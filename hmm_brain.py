"""
Project Regime-Master — HMM Brain
Gaussian Hidden Markov Model for market regime classification.
States: Bull (0), Bear (1), Chop (2), Crash (3)
"""
import numpy as np
import logging
from datetime import datetime
from hmmlearn.hmm import GaussianHMM

import config

logger = logging.getLogger("HMMBrain")

# Feature columns used for HMM training/prediction (must match feature_engine.compute_hmm_features)
HMM_FEATURES = ["log_return", "volatility", "volume_change", "rsi_norm"]


class HMMBrain:
    """
    Wraps hmmlearn.GaussianHMM to classify market regimes.
    
    After training, states are re-ordered by mean log-return:
      - Highest mean  → BULL  (state 0)
      - Moderate neg  → BEAR  (state 1)
      - Near-zero     → CHOP  (state 2)
      - Lowest mean   → CRASH (state 3)
    """

    def __init__(self, n_states=None):
        self.n_states = n_states or config.HMM_N_STATES
        self.model = None
        self._state_map = None        # raw_state → canonical_state
        self._last_trained = None
        self._is_trained = False
        self._feat_mean = None
        self._feat_std = None

    # ─── Training ────────────────────────────────────────────────────────────

    def train(self, df):
        """
        Train HMM on a DataFrame with HMM feature columns.
        
        Parameters
        ----------
        df : pd.DataFrame
            Must contain HMM_FEATURES columns (from feature_engine.compute_hmm_features).
        
        Returns
        -------
        self
        """
        features = df[HMM_FEATURES].dropna().values

        if len(features) < 50:
            logger.warning("Insufficient data for HMM training (%d rows). Need ≥50.", len(features))
            return self

        # Scale features to prevent covariance issues
        self._feat_mean = features.mean(axis=0)
        self._feat_std = features.std(axis=0)
        self._feat_std[self._feat_std < 1e-10] = 1e-10  # avoid div-by-zero
        features_scaled = (features - self._feat_mean) / self._feat_std

        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=config.HMM_COVARIANCE,
            n_iter=config.HMM_ITERATIONS,
            random_state=42,
        )

        self.model.fit(features_scaled)
        self._build_state_map()
        self._last_trained = datetime.utcnow()
        self._is_trained = True

        logger.info(
            "HMM trained on %d samples. State means (log-ret): %s",
            len(features),
            {config.REGIME_NAMES[v]: f"{self.model.means_[k][0]:.6f}"
             for k, v in self._state_map.items()},
        )
        return self

    def _build_state_map(self):
        """
        Map raw HMM states → canonical regime labels by sorting on mean log-return.
        Highest return → BULL, then CHOP (near zero), then BEAR, then CRASH (most negative).
        """
        means = self.model.means_[:, 0]   # log-return means per raw state
        vols  = self.model.means_[:, 1]    # volatility means per raw state

        # Sort states: highest mean first → lowest
        sorted_indices = np.argsort(means)[::-1]

        # If we have 4 states:  [best, ..., worst]
        #   best          → BULL
        #   near-zero     → CHOP  (2nd or 3rd depending on vol)
        #   moderate neg  → BEAR
        #   worst + hi vol→ CRASH
        if self.n_states >= 4:
            # Rank by return: 0=best, 3=worst
            ranked = list(sorted_indices)
            # The two middle states: assign lower-vol one to CHOP, higher-vol to BEAR
            mid = ranked[1:3]
            if vols[mid[0]] <= vols[mid[1]]:
                chop_raw, bear_raw = mid[0], mid[1]
            else:
                chop_raw, bear_raw = mid[1], mid[0]

            self._state_map = {
                ranked[0]:  config.REGIME_BULL,
                bear_raw:   config.REGIME_BEAR,
                chop_raw:   config.REGIME_CHOP,
                ranked[-1]: config.REGIME_CRASH,
            }
        elif self.n_states == 3:
            self._state_map = {
                sorted_indices[0]: config.REGIME_BULL,
                sorted_indices[1]: config.REGIME_CHOP,
                sorted_indices[2]: config.REGIME_BEAR,
            }
        else:
            # 2-state fallback
            self._state_map = {
                sorted_indices[0]: config.REGIME_BULL,
                sorted_indices[1]: config.REGIME_BEAR,
            }

    # ─── Prediction ──────────────────────────────────────────────────────────

    def predict(self, df):
        """
        Predict the CURRENT regime from the latest data.
        
        Returns
        -------
        (canonical_state: int, confidence: float)
        """
        if not self._is_trained:
            logger.warning("HMM not trained yet. Returning CHOP with 0 confidence.")
            return config.REGIME_CHOP, 0.0

        features = df[HMM_FEATURES].dropna().values
        if len(features) == 0:
            return config.REGIME_CHOP, 0.0

        features_scaled = (features - self._feat_mean) / self._feat_std

        raw_state = self.model.predict(features_scaled)[-1]
        probs = self.model.predict_proba(features_scaled)[-1]

        canonical = self._state_map.get(raw_state, config.REGIME_CHOP)
        confidence = float(np.max(probs))

        return canonical, confidence

    def predict_all(self, df):
        """
        Predict regime for entire DataFrame (used by backtester).
        
        Returns
        -------
        np.ndarray of canonical states
        """
        if not self._is_trained:
            return np.full(len(df), config.REGIME_CHOP)

        features = df[HMM_FEATURES].dropna().values
        features_scaled = (features - self._feat_mean) / self._feat_std
        raw_states = self.model.predict(features_scaled)

        # Map raw → canonical
        canonical = np.array([self._state_map.get(s, config.REGIME_CHOP) for s in raw_states])
        return canonical

    def predict_proba_all(self, df):
        """
        Get state probabilities for entire DataFrame.
        
        Returns
        -------
        np.ndarray of shape (n_samples, n_states) — max prob per row = confidence
        """
        if not self._is_trained:
            return np.zeros((len(df), self.n_states))

        features = df[HMM_FEATURES].dropna().values
        features_scaled = (features - self._feat_mean) / self._feat_std
        return self.model.predict_proba(features_scaled)

    # ─── Auto-Retrain ────────────────────────────────────────────────────────

    def needs_retrain(self):
        """Check if the model is stale and needs retraining."""
        if not self._is_trained or self._last_trained is None:
            return True
        hours_since = (datetime.utcnow() - self._last_trained).total_seconds() / 3600
        return hours_since >= config.HMM_RETRAIN_HOURS

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def get_regime_name(self, state):
        """Convert canonical state int → human-readable regime name."""
        return config.REGIME_NAMES.get(state, "UNKNOWN")

    @property
    def is_trained(self):
        return self._is_trained

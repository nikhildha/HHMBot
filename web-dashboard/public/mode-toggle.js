/**
 * SENTINEL — Master Mode Toggle (Paper / Live)
 * Shared across all pages via localStorage.
 * Fires 'mode-change' event on window when toggled.
 */

(function () {
    const STORAGE_KEY = 'sentinelMode';
    const VALID = ['PAPER', 'LIVE'];

    /** Get current mode */
    function getMode() {
        const m = localStorage.getItem(STORAGE_KEY);
        return VALID.includes(m) ? m : 'PAPER';
    }

    /** Set mode and fire event */
    function setMode(mode) {
        if (!VALID.includes(mode)) return;
        localStorage.setItem(STORAGE_KEY, mode);
        applyToggleUI();
        window.dispatchEvent(new CustomEvent('mode-change', { detail: { mode } }));
    }

    /** Toggle between PAPER and LIVE */
    function toggleMode() {
        setMode(getMode() === 'PAPER' ? 'LIVE' : 'PAPER');
    }

    /** Apply visual state to the toggle */
    function applyToggleUI() {
        const mode = getMode();
        const isLive = mode === 'LIVE';

        const thumb = document.getElementById('masterModeThumb');
        const track = document.getElementById('masterModeTrack');
        const labelPaper = document.getElementById('masterLabelPaper');
        const labelLive = document.getElementById('masterLabelLive');

        if (!thumb || !track) return;

        if (isLive) {
            track.style.background = 'rgba(239, 68, 68, 0.15)';
            track.style.borderColor = '#EF4444';
            thumb.style.background = '#EF4444';
            thumb.style.left = '27px';
            if (labelPaper) labelPaper.style.color = 'var(--text-tertiary, #8899AA)';
            if (labelLive) labelLive.style.color = '#EF4444';
        } else {
            track.style.background = 'rgba(34, 197, 94, 0.15)';
            track.style.borderColor = '#22C55E';
            thumb.style.background = '#22C55E';
            thumb.style.left = '3px';
            if (labelPaper) labelPaper.style.color = '#22C55E';
            if (labelLive) labelLive.style.color = 'var(--text-tertiary, #8899AA)';
        }
    }

    /** Inject toggle HTML into header-right */
    function injectToggle() {
        const headerRight = document.querySelector('.header-right');
        if (!headerRight) return;

        // Don't inject twice
        if (document.getElementById('masterModeToggle')) {
            applyToggleUI();
            return;
        }

        const toggle = document.createElement('div');
        toggle.id = 'masterModeToggle';
        toggle.className = 'master-mode-toggle';
        toggle.innerHTML = `
            <span id="masterLabelPaper" class="mode-label">PAPER</span>
            <div id="masterModeTrack" class="mode-track" onclick="window.sentinelToggleMode()">
                <div id="masterModeThumb" class="mode-thumb"></div>
            </div>
            <span id="masterLabelLive" class="mode-label">LIVE</span>
        `;

        // Insert at the beginning of header-right
        headerRight.insertBefore(toggle, headerRight.firstChild);
        applyToggleUI();
    }

    // Expose to window
    window.sentinelGetMode = getMode;
    window.sentinelSetMode = setMode;
    window.sentinelToggleMode = toggleMode;

    // Auto-inject on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', injectToggle);
    } else {
        injectToggle();
    }

    // Listen for changes from other tabs
    window.addEventListener('storage', (e) => {
        if (e.key === STORAGE_KEY) {
            applyToggleUI();
            window.dispatchEvent(new CustomEvent('mode-change', { detail: { mode: getMode() } }));
        }
    });
})();

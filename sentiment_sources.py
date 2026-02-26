"""
Project Regime-Master — Sentiment Sources
Individual data-source adapters for news, social, and sentiment indicators.

Sources (all free-tier friendly):
  1. CryptoPanicSource  — Aggregated crypto news API (cryptopanic.com)
  2. RSSNewsSource       — CoinTelegraph, Decrypt, CoinDesk, The Block, Bitcoin Magazine
  3. RedditSource        — r/cryptocurrency + coin-specific subreddits (requires PRAW keys)
  4. FearGreedSource     — Alternative.me Fear & Greed Index (no key needed)

Each source returns a list[ArticleItem].  Missing optional deps or network errors
produce empty lists — never raise to callers.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import List, Optional

import requests

import config

logger = logging.getLogger("SentimentSources")

# ─── Coin keyword / subreddit mappings ───────────────────────────────────────

# Maps normalised coin symbol → list of text keywords to search in articles
COIN_KEYWORDS: dict[str, list[str]] = {
    "BTC":   ["bitcoin", "btc", "satoshi"],
    "ETH":   ["ethereum", "eth", "ether"],
    "BNB":   ["bnb", "binance coin", "binance smart chain", "bsc"],
    "SOL":   ["solana", "sol"],
    "XRP":   ["xrp", "ripple"],
    "ADA":   ["cardano", "ada"],
    "DOGE":  ["dogecoin", "doge"],
    "MATIC": ["polygon", "matic"],
    "DOT":   ["polkadot", "dot"],
    "AVAX":  ["avalanche", "avax"],
    "LINK":  ["chainlink", "link"],
    "ATOM":  ["cosmos", "atom"],
    "UNI":   ["uniswap", "uni"],
    "NEAR":  ["near protocol", "near"],
    "LTC":   ["litecoin", "ltc"],
    "BCH":   ["bitcoin cash", "bch"],
    "TRX":   ["tron", "trx"],
    "SHIB":  ["shiba inu", "shib"],
    "FTM":   ["fantom", "ftm"],
    "ALGO":  ["algorand", "algo"],
    "VET":   ["vechain", "vet"],
    "ICP":   ["internet computer", "icp"],
    "MANA":  ["decentraland", "mana"],
    "SAND":  ["the sandbox", "sand"],
    "AXS":   ["axie infinity", "axs"],
    "AAVE":  ["aave"],
    "GRT":   ["the graph", "grt"],
    "MKR":   ["maker", "mkr", "makerdao"],
    "COMP":  ["compound", "comp"],
    "SNX":   ["synthetix", "snx"],
    "CRV":   ["curve", "crv"],
    "SUSHI": ["sushiswap", "sushi"],
    "1INCH": ["1inch"],
    "FIL":   ["filecoin", "fil"],
    "EOS":   ["eos"],
    "XLM":   ["stellar", "xlm"],
    "XMR":   ["monero", "xmr"],
    "ZEC":   ["zcash", "zec"],
    "DASH":  ["dash"],
    "NEO":   ["neo"],
    "ONT":   ["ontology", "ont"],
    "ZIL":   ["zilliqa", "zil"],
    "THETA": ["theta"],
    "ENJ":   ["enjin", "enj"],
    "CHZ":   ["chiliz", "chz"],
    "BAT":   ["basic attention token", "bat"],
    "ZRX":   ["0x", "zrx"],
    "STORJ": ["storj"],
    "KAVA":  ["kava"],
    "BAND":  ["band protocol", "band"],
    "REN":   ["ren protocol", "ren"],
}

# Reddit subreddits per coin (used when PRAW is configured)
COIN_SUBREDDITS: dict[str, str] = {
    "BTC":   "Bitcoin",
    "ETH":   "ethereum",
    "BNB":   "binance",
    "SOL":   "solana",
    "XRP":   "Ripple",
    "ADA":   "cardano",
    "DOGE":  "dogecoin",
    "MATIC": "maticnetwork",
    "DOT":   "dot",
    "AVAX":  "avax",
    "LINK":  "chainlink",
    "ATOM":  "cosmosnetwork",
    "UNI":   "Uniswap",
    "NEAR":  "nearprotocol",
    "LTC":   "litecoin",
    "SHIB":  "SHIBArmy",
    "AAVE":  "Aave",
    "MKR":   "MakerDAO",
    "FIL":   "filecoin",
    "EOS":   "eos",
    "XLM":   "Stellar",
    "XMR":   "Monero",
}

# ─── Data model ──────────────────────────────────────────────────────────────

@dataclass
class ArticleItem:
    """One news/social item from any source."""
    title: str
    body: str                           # may be empty
    source: str                         # e.g. "CryptoPanic", "Reddit", "RSS"
    coins: List[str]                    # normalised symbols, e.g. ["BTC", "ETH"]
    published_at: datetime              # always UTC-aware
    url: str = ""
    importance: float = 0.5            # 0-1; source-specific quality signal
    tags: List[str] = field(default_factory=list)  # e.g. ["bullish", "important"]

    @property
    def text(self) -> str:
        """Combined title + body for NLP."""
        return (self.title + " " + self.body).strip()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(value) -> Optional[datetime]:
    """Robustly parse a datetime from string, struct_time, or datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    # struct_time (from feedparser)
    try:
        import calendar
        ts = calendar.timegm(value)
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except Exception:
        pass
    # ISO string
    try:
        s = str(value).rstrip("Z")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _coin_mentions(text: str) -> List[str]:
    """Return list of coin symbols mentioned in text (lowercase match)."""
    text_lower = text.lower()
    found = []
    for symbol, keywords in COIN_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            found.append(symbol)
    return found


def _strip_html(text: str) -> str:
    """Strip HTML tags from text."""
    return re.sub(r"<[^>]+>", " ", text or "").strip()


# ─── Base class ──────────────────────────────────────────────────────────────

class BaseSource:
    """Abstract news source. fetch() never raises; returns [] on any error."""

    name: str = "Base"

    def fetch(self, coins: List[str], since: datetime) -> List[ArticleItem]:
        raise NotImplementedError


# ─── 1. CryptoPanic ──────────────────────────────────────────────────────────

class CryptoPanicSource(BaseSource):
    """
    CryptoPanic aggregates 100+ crypto news sources.
    Free tier: up to 50 posts per request (use auth_token for more).
    API docs: https://cryptopanic.com/developers/api/
    """
    name = "CryptoPanic"
    _BASE = "https://cryptopanic.com/api/v1/posts/"
    _RATE_LIMIT_SLEEP = 1.2  # seconds between paginated requests

    def fetch(self, coins: List[str], since: datetime) -> List[ArticleItem]:
        try:
            return self._fetch_impl(coins, since)
        except Exception as e:
            logger.warning("[CryptoPanic] fetch failed: %s", e)
            return []

    def _fetch_impl(self, coins: List[str], since: datetime) -> List[ArticleItem]:
        articles: List[ArticleItem] = []
        # CryptoPanic accepts up to ~10 currencies per request
        currency_str = ",".join(c for c in coins if c in COIN_KEYWORDS)[:80]

        params = {
            "public":    "true",
            "kind":      "news",
            "regions":   "en",
            "currencies": currency_str or None,
        }
        if config.CRYPTOPANIC_API_KEY:
            params["auth_token"] = config.CRYPTOPANIC_API_KEY

        # Paginate up to 3 pages
        url = self._BASE
        for page in range(1, 4):
            try:
                resp = requests.get(url, params=params if page == 1 else None,
                                    timeout=10)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.debug("[CryptoPanic] page %d failed: %s", page, e)
                break

            results = data.get("results", [])
            for item in results:
                pub = _parse_dt(item.get("published_at"))
                if pub is None or pub < since:
                    return articles  # results are newest-first; stop when too old

                raw_coins = [c["code"].upper() for c in item.get("currencies", [])]
                tags_raw = item.get("votes", {})
                tags = []
                if tags_raw.get("positive", 0) > tags_raw.get("negative", 0):
                    tags.append("bullish")
                elif tags_raw.get("negative", 0) > tags_raw.get("positive", 0):
                    tags.append("bearish")

                # Importance: "important" and "hot" labels boost the signal
                importance = 0.5
                if item.get("kind") == "news":
                    importance = 0.7
                for label in item.get("currencies", []):
                    if label.get("important"):
                        importance = min(1.0, importance + 0.2)

                articles.append(ArticleItem(
                    title=item.get("title", ""),
                    body="",
                    source=self.name,
                    coins=raw_coins,
                    published_at=pub,
                    url=item.get("url", ""),
                    importance=importance,
                    tags=tags,
                ))

            # Get next page URL
            next_url = data.get("next")
            if not next_url or not results:
                break
            url = next_url
            params = {}  # params already baked into next_url
            time.sleep(self._RATE_LIMIT_SLEEP)

        logger.debug("[CryptoPanic] fetched %d articles", len(articles))
        return articles


# ─── 2. RSS News ─────────────────────────────────────────────────────────────

class RSSNewsSource(BaseSource):
    """
    Parses RSS feeds from major crypto media outlets.
    Requires: pip install feedparser
    """
    name = "RSS"

    def fetch(self, coins: List[str], since: datetime) -> List[ArticleItem]:
        try:
            import feedparser
        except ImportError:
            logger.warning("[RSS] feedparser not installed. Run: pip install feedparser")
            return []

        articles: List[ArticleItem] = []
        for feed_url in config.SENTIMENT_RSS_FEEDS:
            try:
                feed = feedparser.parse(feed_url)
                source_name = feed.feed.get("title", feed_url.split("/")[2])
                for entry in feed.entries:
                    pub = _parse_dt(entry.get("published_parsed") or entry.get("updated_parsed"))
                    if pub is None:
                        continue
                    if pub < since:
                        continue  # too old

                    title = _strip_html(entry.get("title", ""))
                    summary = _strip_html(entry.get("summary", "") or entry.get("content", [{}])[0].get("value", ""))
                    combined = title + " " + summary
                    coins_in_article = _coin_mentions(combined)
                    if not coins_in_article:
                        # Still include global market articles — tag as "CRYPTO"
                        coins_in_article = ["CRYPTO"]

                    articles.append(ArticleItem(
                        title=title,
                        body=summary[:500],  # cap body length
                        source=f"RSS:{source_name}",
                        coins=coins_in_article,
                        published_at=pub,
                        url=entry.get("link", ""),
                        importance=0.6,
                    ))
            except Exception as e:
                logger.debug("[RSS] feed %s failed: %s", feed_url, e)

        logger.debug("[RSS] fetched %d articles from %d feeds",
                     len(articles), len(config.SENTIMENT_RSS_FEEDS))
        return articles


# ─── 3. Reddit ───────────────────────────────────────────────────────────────

class RedditSource(BaseSource):
    """
    Reads hot/new posts from r/cryptocurrency and coin-specific subreddits.

    Requires:
      pip install praw
      REDDIT_CLIENT_ID + REDDIT_CLIENT_SECRET in .env
      (Create an app at reddit.com/prefs/apps — choose 'script' type)

    Falls back to Reddit's public RSS feed if PRAW is not configured.
    """
    name = "Reddit"
    _FALLBACK_RSS = "https://www.reddit.com/r/CryptoCurrency/hot.json?limit=50"

    def fetch(self, coins: List[str], since: datetime) -> List[ArticleItem]:
        if config.REDDIT_CLIENT_ID:
            return self._fetch_praw(coins, since)
        return self._fetch_public_json(since)

    def _fetch_praw(self, coins: List[str], since: datetime) -> List[ArticleItem]:
        try:
            import praw
        except ImportError:
            logger.warning("[Reddit] praw not installed. Run: pip install praw")
            return self._fetch_public_json(since)

        try:
            reddit = praw.Reddit(
                client_id=config.REDDIT_CLIENT_ID,
                client_secret=config.REDDIT_CLIENT_SECRET,
                user_agent=config.REDDIT_USER_AGENT,
            )
            articles: List[ArticleItem] = []
            subreddits_to_fetch = ["CryptoCurrency"]

            for coin in coins:
                sub = COIN_SUBREDDITS.get(coin)
                if sub and sub not in subreddits_to_fetch:
                    subreddits_to_fetch.append(sub)

            for sub_name in subreddits_to_fetch[:8]:  # cap to avoid rate limits
                try:
                    sub = reddit.subreddit(sub_name)
                    for post in sub.hot(limit=30):
                        pub = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)
                        if pub < since:
                            continue
                        text = (post.title or "") + " " + (post.selftext or "")[:300]
                        coins_in = _coin_mentions(text) or ["CRYPTO"]
                        importance = min(1.0, 0.4 + min(post.score, 5000) / 10000)
                        articles.append(ArticleItem(
                            title=post.title,
                            body=(post.selftext or "")[:300],
                            source=f"Reddit:r/{sub_name}",
                            coins=coins_in,
                            published_at=pub,
                            url=f"https://reddit.com{post.permalink}",
                            importance=importance,
                        ))
                except Exception as e:
                    logger.debug("[Reddit/PRAW] subreddit %s failed: %s", sub_name, e)

            logger.debug("[Reddit/PRAW] fetched %d posts", len(articles))
            return articles
        except Exception as e:
            logger.warning("[Reddit/PRAW] failed: %s", e)
            return self._fetch_public_json(since)

    def _fetch_public_json(self, since: datetime) -> List[ArticleItem]:
        """Fallback: Reddit's public .json endpoint (no auth required)."""
        articles: List[ArticleItem] = []
        try:
            headers = {"User-Agent": config.REDDIT_USER_AGENT}
            resp = requests.get(self._FALLBACK_RSS, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            posts = data.get("data", {}).get("children", [])
            for child in posts:
                p = child.get("data", {})
                pub = datetime.fromtimestamp(p.get("created_utc", 0), tz=timezone.utc)
                if pub < since:
                    continue
                title = p.get("title", "")
                body  = (p.get("selftext", "") or "")[:300]
                coins_in = _coin_mentions(title + " " + body) or ["CRYPTO"]
                importance = min(1.0, 0.4 + min(p.get("score", 0), 5000) / 10000)
                articles.append(ArticleItem(
                    title=title,
                    body=body,
                    source="Reddit:r/CryptoCurrency",
                    coins=coins_in,
                    published_at=pub,
                    url=f"https://reddit.com{p.get('permalink', '')}",
                    importance=importance,
                ))
        except Exception as e:
            logger.debug("[Reddit/JSON] fetch failed: %s", e)
        logger.debug("[Reddit/JSON] fetched %d posts", len(articles))
        return articles


# ─── 4. Fear & Greed Index ───────────────────────────────────────────────────

@dataclass
class FearGreedReading:
    """Single Fear & Greed snapshot."""
    score: int           # 0 (Extreme Fear) – 100 (Extreme Greed)
    label: str           # e.g. "Greed", "Fear"
    timestamp: datetime
    normalized: float    # (score - 50) / 50  →  -1 to +1


class FearGreedSource:
    """
    Alternative.me Crypto Fear & Greed Index.
    No API key required.  Updates once per day.
    """
    _API_URL = "https://api.alternative.me/fng/?limit=2&format=json"

    def fetch(self) -> Optional[FearGreedReading]:
        try:
            resp = requests.get(self._API_URL, timeout=8)
            resp.raise_for_status()
            data = resp.json()
            latest = data["data"][0]
            score = int(latest["value"])
            label = latest["value_classification"]
            ts = datetime.fromtimestamp(int(latest["timestamp"]), tz=timezone.utc)
            return FearGreedReading(
                score=score,
                label=label,
                timestamp=ts,
                normalized=(score - 50) / 50.0,
            )
        except Exception as e:
            logger.warning("[FearGreed] fetch failed: %s", e)
            return None

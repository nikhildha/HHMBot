import sys
try:
    import feedparser
    print(f"Success: feedparser imported from {feedparser.__file__}")
except ImportError as e:
    print(f"Failure: {e}")
    print(f"Sys Path: {sys.path}")

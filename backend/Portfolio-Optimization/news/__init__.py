"""Per-stock news and social sentiment (Phase 2).

    keywords.py     a tiered keyword profile per held symbol (company / sector / macro)
    channels/       one adapter per source, all searched with the same profile
    relevance.py    is an item actually about this stock (or its sector / economy)?
    scoring.py      polarity per item, by language
    aggregate.py    per-tier sentiment, momentum and attention-spike features
    pipeline.py     the active-set preparation step that ties it together

Only aggregates and source links are persisted. Raw social posts are never stored.
"""

"""Per-exchange adapters: CSE, Binance and Yahoo Finance.

Each adapter answers the same two questions -- "what instruments exist" and "give me daily
bars for one of them" -- and normalises the answer to data.ingestion.OHLCV_COLUMNS, so
everything above this package is exchange-agnostic.
"""

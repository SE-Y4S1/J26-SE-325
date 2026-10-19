"""Fundamental analysis for held symbols (Phase 2b).

  reports.py    where statements come from: CSE report archive (PDF), yfinance, CoinGecko
  extract.py    CSE PDF text -> canonical statement fields, with consistency checks
  ratios.py     ratios, Piotroski F-score, Altman Z''-score, sector-relative percentiles
  redflags.py   going-concern / audit / contingency flags and outlook tone from report text
  pipeline.py   the active-set step and the point-in-time store
"""

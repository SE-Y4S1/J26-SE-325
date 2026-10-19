"""Phase 4b: the Laya decision model for liquidation.

  questions.py   the typed questions (Stage 1 analysis, Stage 2 decision)
  states.py      compact per-holding states built from the prepared snapshot (<= 512 tokens)
  engine.py      loading Laya (fine-tuned > base > unavailable), asking, applying to the plan
  labels.py      outcome-based training labels and the JSONL dataset for laya.train.finetune
  checkpoint.py  head-only export (Colab) and local install over the cached base checkpoint

Laya never produces a number the plan uses directly. It answers typed questions with
calibrated probabilities; the GA still computes every amount. Answers below the confidence
threshold are not used, and the fuzzy rules decide alone -- recorded per holding.
"""

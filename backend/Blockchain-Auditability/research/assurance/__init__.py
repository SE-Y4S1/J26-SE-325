"""IEEE-CIS evaluation of Component 3's AI Decision Assurance Engine.

Pipeline: data (time-ordered splits) -> features (fit on train) -> models (fit on train,
calibrated on the calibration split) -> signals (label-free) -> routers (tuned on the
policy-tuning split) -> evaluate (held-out test split, once) -> export (the tuned P002 policy
and golden vectors for the JavaScript engine).
"""

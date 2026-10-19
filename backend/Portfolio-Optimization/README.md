# Component 1 — Liquidity-Aware Forecasting & Portfolio Optimization Engine

Part of **J26-SE-325**, "An Integrated AI-Driven Smart Finance Platform" (SLIIT IT4010).

> **Sub-objective (TAF):** Design a liquidity-aware forecasting and portfolio optimization
> engine that supports instant, loss-minimized user withdrawals alongside long-term return
> optimization.

> **Research gap this closes (TAF §6):** *"Liquidity-aware withdrawal planning has not been
> operationalized as a real-time, user-facing service."*

Commercial robo-advisors automate long-term allocation with mean-variance optimization but
offer no instant, loss-minimized withdrawal mechanism. This component builds one, and
measures whether it actually beats the alternatives.

---

## Research questions

Each has a concrete metric behind it. This is what separates a pipeline that runs from a
defensible result.

| RQ | Question | Metric | Comparison |
|----|----------|--------|------------|
| **RQ1** | Does the fine-tuned foundation + LSTM hybrid beat a plain LSTM and a zero-shot foundation model? | MAE, RMSE, pinball loss, quantile coverage | LSTM baseline vs zero-shot vs LoRA-tuned vs hybrid — split by asset class |
| **RQ2** | Does the MOEA/D liquidity-aware optimizer beat classic mean-variance at equal or lower realized cost? | Sharpe, Sortino, max drawdown, realized transaction cost | Markowitz vs MOEA/D, across 3 Pareto selection rules |
| **RQ3** | Does the fuzzy-GA withdrawal module reduce realized slippage vs naive liquidation? | Realized slippage %, realized loss | Fuzzy GA vs pro-rata / largest-first / most-liquid-first |
| **RQ4** | How does withdrawal-plan quality degrade as liquidity worsens, and where does it break? | Degradation curve, breakdown point | RQ3 repeated across a stress-severity ladder |

RQ1–RQ4 are **not** written in the TAF — they come from the build brief and are stated here
so they become the documented evaluation frame.

---

## Architecture

```
ingestion → features → forecasting → optimization → service
  yfinance   pandas-ta    TimesFM /      MOEA/D (long-term)      FastAPI
  GDELT      FinBERT      Chronos-Bolt   fuzzy GA (withdrawal)   Kafka
  NewsAPI                 + residual head                        events
                                ↑                    ↑
                                └──── agent tools ───┘
                                  (Ollama gemma4-e4b →
                                   fine-tuned Llama)
```

Two **separate** optimizers, per the TAF's *"MOEA/D, fuzzy genetic algorithm"*:

- **MOEA/D** (`optimization/moead_rebalance.py`) — long-term allocation over the weight
  simplex. 3 objectives: maximize return, minimize CVaR, minimize liquidity cost.
- **Fuzzy GA** (`optimization/fuzzy_withdrawal.py` + `ga_withdrawal.py`) — instant
  liquidation planning. A fuzzy inference system scores per-holding sell priority; a GA
  searches ordered (asset, fraction) schedules under a hard "raise $X within N days"
  constraint.

They are not interchangeable and are never merged. Different decision variables, different
constraints, different algorithms.

### The hybrid forecaster

```
final_forecast = base_forecast + residual_head(technical, sentiment, base_forecast, asset_class)
```

A foundation model pretrained on raw univariate series captures trend and seasonality well
but has no channel for exogenous features. The residual head learns the systematic part of
its error as a function of those features. The base stays frozen (or LoRA-adapted) and the
head is zero-initialized, so the hybrid starts exactly at the base model's accuracy — which
gives RQ1 a clean answer to "did the covariates help".

**How the head's training data is generated.** The head learns from base forecasts produced
by walking a cut point forward: the base predicts from `group[:t+1]`, and that forecast is
stamped at `t`, where it lines up with the `target_return` it is actually predicting. The
base never sees its own target.

This matters more than it sounds. Both foundation adapters return a *single* row per call,
stamped at the last input timestamp — and under `add_targets(horizon=h)` that row's target
is always NaN, because the final `h` rows of every symbol have no realised future yet. One
call per symbol therefore produced exactly the rows that cannot be trained on, and the
hybrid failed with `no overlap between base forecasts and targets` on **every** foundation
base. It passed with the LSTM only because that adapter returns many in-sample rows, which
is why no test caught it until Colab did. `tests/test_forecasting.py` now carries a
single-row stub base with the foundation adapters' shape.

Each cut costs one forward pass, so `HybridConfig.walk_forward_points` caps how many a
symbol is worth; `min_context` sets the shortest prefix worth forecasting from. Early cuts
give the base less context than it would have in production, which is inherent to
walk-forward generation rather than a defect — but it is worth stating when reporting RQ1.


### The grounding constraint

The agent may reason over signals and decide *how* to invoke the optimizer. It may **never**
emit a sell amount that did not come from `run_fuzzy_ga_withdrawal`.

This is enforced in code (`agent/reference_agent.py::enforce_grounding`), not by convention:
a transcript is valid only if the tool was called **and** every number in the final decision
matches that call's output. Requiring only the call would let a model invoke the tool, ignore
it, and invent its own figures.

The reason is regulatory, from the TAF's Legal Impact section: *"Financial regulators
increasingly mandate explainability, auditability, and bias mitigation… platforms lacking
tamper-evident audit trails and human-override governance face compliance exposure."* An LLM
producing "sell 40% of AAPL" from raw signals has no auditable derivation. The fuzzy-GA
output does, complete with a rule trace.

### Scope boundary with Component 4

This component's agent decides **which** strategy to execute and returns a structured
decision plus a short internal reasoning trace. Component 4's agent explains **why**, to end
users, via SHAP/LIME and the Trust Panel. No user-facing prose is produced here.

---

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

### One command

```bash
uv run python run_pipeline.py --list     # what is done, what is pending
uv run python run_pipeline.py            # everything runnable offline
```

Stages are skipped when their output already exists, so re-running is cheap and a partial
failure resumes rather than restarts. Fine-tuning and RQ1's foundation/hybrid rows are
deliberately excluded -- they need a GPU and a fast link, so they live in
`experiments/colab_finetune.ipynb`.

### From scratch

```bash
cd backend/Portfolio-Optimization
uv venv --python 3.12

# torch FIRST, from the CPU index. The default Windows PyPI wheel bundles ~2.5GB of CUDA
# libraries that an AMD/Intel machine can never use, and it dominates install time.
uv pip install --index-url https://download.pytorch.org/whl/cpu torch

uv pip install -r requirements.txt
uv run pytest tests/test_env.py -v      # dependency gate — run this first
```

`uv sync --group dev` also works and is the cleaner route, but it resolves torch from PyPI
unless you keep the `[[tool.uv.index]]` block in `pyproject.toml` — and on a slow connection
it downloads everything before linking anything, so failures surface late. The staged form
above fails fast and was what actually worked on the development machine.

Optional foundation models (either or both; ~821MB of weights each, downloaded on first use):

```bash
uv pip install "timesfm[torch]==2.0.2"      # note: 2.0.2, NOT 2.5 — see deviations table
uv pip install chronos-forecasting

# Weights are NOT fetched automatically. Only run these if you need RQ1's foundation-model
# comparison; each is ~821MB. See "Model policy" below.
huggingface-cli download amazon/chronos-bolt-base
huggingface-cli download google/timesfm-2.5-200m-pytorch
```

Without those weights the pipeline still runs end to end: `baseline_lstm` covers forecasting
and the local Ollama model covers sentiment.

### The capability report is a required step, not a test artefact

`forecasting/base.py` decides which foundation adapters to register by reading
`artifacts/env_report.json`. It does not import the packages to find out — one broken
optional dependency would otherwise take down the whole registry at import time, making the
LSTM baseline and the entire optimization path unreachable over something neither needs.

`artifacts/` is gitignored, so **a fresh checkout has no report and sees no foundation
models at all**, however the install went. Regenerate it after installing or changing either
optional package:

```bash
uv run python -m forecasting.env_report
```

Skipping this produces the most misleading error in the component — `get_forecaster` raising
*"Chronos-Bolt is not installed"* about a package that is installed and importable. The
message now distinguishes a missing report from a missing package from a package that fails
to import, and prints the underlying error in the last case. The notebook bootstrap runs
this automatically; a fresh clone driven from a shell has to run it.


Copy `.env.example` to `.env` for API keys (all optional — GDELT and yfinance need none).

The reference agent needs [Ollama](https://ollama.com) with `gemma4-e4b` pulled. It is a
stand-in for the Colab-fine-tuned Llama and is confined to `agent/reference_agent.py`.

---

## Deviations from the build brief

Verified against primary sources on 2026-08-24. Each of these would have caused a build
failure if followed as written.

| Brief said | Reality | What we do |
|---|---|---|
| Python 3.10, "required for timesfm" | timesfm is `>=3.10`; `pandas-ta` and `jax` are `>=3.12` | **Python 3.12** — the only version satisfying the whole stack |
| `timesfm` v2.5 | PyPI tops out at **2.0.2**; "2.5" is the *model* generation | Pin `timesfm==2.0.2`, use `TimesFM_2p5_200M_torch` |
| Use TimesFM XReg for covariates | `timesfm[xreg]` → `jax[cuda]` → `jax-cuda12-plugin`, which is **manylinux-only** | Covariates go through the residual head — architecturally cleaner anyway |
| `pandas-ta` | 0.3.14b0 was pulled from PyPI; current release hard-pins `numba==0.61.2` | `pandas-ta-openbb` (same import name, no numba pin) |
| Alpha Vantage for forex | — | yfinance FX tickers: deeper history, no key, same code path as equities |
| NewsAPI for sentiment | Free tier caps at ~30 days — cannot support a walk-forward backtest | GDELT for history + NewsAPI for the live window |
| Hosted API for the reference agent | — | Local Ollama `gemma4-e4b` (verified `tools` capable) — free, offline, reproducible |
| One global data window | — | Per-symbol windows from explicit criteria (`data/window_selector.py`) |

**Added from the TAF, missing from the brief:** *"containerize and monitor the services"* —
Phase 8.

### Fine-tuning the two foundation models

Neither exposes a HuggingFace-style training interface, and they do not share one with each
other, so `forecasting/finetune_lora.py` gives each its own step (`STEP_FUNCTIONS`).

**Chronos-Bolt** computes its own pinball loss: `forward(context, target=...)` normalises the
target with the same loc/scale as the context and pads a short target with a zero mask, so a
5-step horizon against its fixed `prediction_length` needs nothing from us. Scoring its
quantile predictions externally would have applied the wrong normalisation.

**TimesFM 2.5** has no training entry point at all. `decode()` runs under `torch.no_grad()`,
but it is not the only path: `forward()` is an ordinary differentiable module, and for a
horizon within one `output_patch_len` (128) `decode()`'s autoregressive loop never runs, so
its prefill — patch, running-stats normalise, forward, denormalise — *is* the whole
computation. `_timesfm_step` mirrors that prefill line for line, because any divergence is a
train/serve mismatch nothing downstream would catch.

Two traps worth knowing if you touch this:

- The output head emits `len(quantiles) + 1` columns and **column 0 is the mean**. Reading it
  as the 0.1 quantile trains against the wrong target and still converges.
- Every transformer norm `scale` initialises to exactly **zero**, so a randomly-initialised
  TimesFM is an identity function and gradients reach only the tokenizer and output head.
  That looks identical to a broken training step. The slow test fills the scales before
  asserting that gradient reaches the attention projections.

---

## Open items for supervisor sign-off

1. **CVaR is not named anywhere in the TAF.** The form commits only to "liquidity-aware" and
   "loss-minimized". CVaR is a deliberate strengthening (coherent, tail-focused, unlike
   variance) — be ready to justify it.
2. **Forex is not in the TAF's Component 1 dataset line**, which says *"liquid equities
   and/or ETFs"*. Adding FX is a superset; worth a sentence in the methodology.
3. **RQ1–RQ4 are not in the TAF.** They originate in the build brief.
4. **If TimesFM fails the Phase 0 gate**, RQ1 runs on Chronos-Bolt alone — a deviation from a
   TAF-named method. The dual-adapter design is the hedge.
5. The TAF cites a *"Section 13 (Risks, Ethics, and Mitigation)"* and an *"RQ4"* that do not
   exist in the 12-page document — dangling references worth fixing in the next revision.

---

## Empirical findings so far

RQ3 and RQ4 run today without a trained forecaster (`uv run python experiments/run_rq_analysis.py --rq 3 4`).

**RQ3 — the fuzzy GA beats every naive baseline** (mean relative reduction in realized loss,
swept over 3 withdrawal sizes x 3 urgency levels):

| Baseline | Relative improvement |
|---|---|
| pro-rata | **96.5%** |
| largest-position-first | **26.2%** |
| most-liquid-first | **8.7%** |

The ordering is the expected one: most-liquid-first is the strongest baseline because it is
already liquidity-aware, just myopically so.

**Phase 1 validated on real market data.** `configs/resolved_universe.yaml` is generated
by `data/window_selector.py` from live yfinance history for **all 26 configured symbols**.
Reaching 26/26 took several passes: yfinance throttling initially cost SPY, XOM, BA, TLT, XLE
and two FX pairs, which is what motivated the retry-with-backoff and the
`cache_empty=False` rule. The per-symbol criteria demonstrably discriminate rather than
returning a constant:

| Evidence | Observed |
|---|---|
| Availability criterion | **META resolves to 2012-05-18**, its actual IPO date, while every other symbol takes the full 15-year cap |
| Signal-driven horizon | ACF decay lag ranges **1 to 5** across symbols (JPM and SPY at 5; AMZN, CAT, GLD at 1) |
| Resulting horizons | `[1, 5]` where the lag supports a 5-day forecast, `[1]` where it does not |
| Regime coverage | all 24 windows span **3** volatility regimes, satisfying `min_regimes` |
| Liquidity floor | no symbol flagged thin at a 50k position — correct for this large-cap universe |

Each entry records the criterion values that produced it, so no window in the dissertation
is an unexplained constant.

**RQ1 — baseline row, measured on real data** (`baseline_lstm`, horizon 5, walk-forward over
22 symbols, 4 non-overlapping folds, 11,726 out-of-sample predictions):

| Metric | Value |
|---|---|
| MAE | 0.028822 |
| RMSE | 0.038558 |
| Pinball loss | **0.009388** |

Pinball is the number the hybrid must beat; MAE/RMSE only score the median and say nothing
about whether the p10–p90 band is honest.

**The calibration is the interesting part, and it is a finding rather than a defect:**

| Quantile | Observed coverage | Nominal |
|---|---|---|
| p10 | 0.0695 | 0.10 |
| p50 | 0.4513 | 0.50 |
| p90 | 0.8598 | 0.90 |

All three sit *below* nominal by a similar margin. That is a uniform LEVEL bias, not an
interval-width problem: the whole predicted distribution is shifted down, so the model
systematically under-predicts returns. The evaluation window (2021-10 to 2025-12) is
predominantly a rising market, and a model trained on trailing two-year windows will lag a
sustained uptrend — exactly this signature.

**Why this matters downstream, and it is worth a sentence in the methodology:** Phase 5a's
CVaR is computed from these quantiles. An under-predicting forecaster overstates downside
risk, so the MOEA/D optimizer becomes *more conservative than the data warrants*. That is the
safe direction to be wrong in, but it is a bias, and RQ2's allocations inherit it. A hybrid
that corrects the level bias should therefore improve RQ2 as well as RQ1 — a testable
prediction the residual head is well placed to deliver, since a constant offset is the
easiest thing for it to learn.

**Hybrid fusion validated on real data, before Colab.** The TAF's hybrid needs TimesFM,
whose weights are unreachable here — but the *fusion machinery* does not care which
Forecaster is underneath. Running `HybridForecaster(base=baseline_lstm)` on real AAPL/MSFT/SPY
history exercised the whole path locally and caught a genuine bug: `fit()` assumed the base
was already usable, which holds for a zero-shot adapter (its `fit` is a no-op) but breaks for
any *trainable* base. That would have failed inside a Colab run rather than here.

With it fixed, the decomposition shows the residual head doing precisely its job
(5-day horizon, target std 0.0309):

| Quantile | Base | Residual | Final |
|---|---|---|---|
| p10 | −0.17395 | **+0.14437** | −0.02957 |
| p50 | +0.00197 | +0.00299 | +0.00496 |
| p90 | +0.05680 | −0.02258 | +0.03422 |

The base LSTM emits an implausible −17% five-day p10; the head narrows it to −2.96%, which is
realistic against a 3.1% target standard deviation. This is the residual-correction rationale
demonstrated rather than merely asserted — and it is consistent with the calibration finding
above, since correcting a systematically mis-scaled interval is exactly what the head is for.

**Compute budget is a stated choice, not an accident.** The full 15-year universe at
a six-month step yields ~23 walk-forward folds, each training an LSTM from scratch: roughly a
quarter of a million gradient steps, i.e. hours on a CPU-only laptop. `scripts_rq1.py`
therefore evaluates a 5-year span with annual steps (~5 folds) and a smaller network.

That is a *smaller* experiment, not a *weaker* one. Folds remain non-overlapping and the
`embargo_days >= horizon` rule still holds, so every reported figure is an honest
out-of-sample estimate; there is simply less of it. Widen `EVAL_YEARS` and raise the epoch
count when running with a GPU. Report the fold count alongside the metrics -- five folds
supports "the hybrid beats the baseline here", not a significance claim.

**RQ2 — MOEA/D matches Markowitz's return at a fraction of the cost:**

| Method | Expected return | Liquidity cost | THIN weight |
|---|---|---|---|
| Markowitz (max-Sharpe) | 0.000678 | 0.000083 | 0.0897 |
| MOEA/D (knee) | 0.000650 | 0.000081 | 0.0836 |
| **MOEA/D (max-Sharpe)** | **0.000678** | **0.000024** | 0.0969 |
| MOEA/D (scalarized) | 0.000678 | 0.000024 | 0.0969 |

Identical expected return at **3.5x lower liquidity cost** — precisely the "equal or lower
realized transaction cost" RQ2 asks for. The mechanism is worth stating in the write-up
because it is counter-intuitive: MOEA/D *holds more* of the illiquid name (0.0969 vs 0.0897)
yet pays less, because liquidity cost is charged on the **trade**, not the holding. Starting
from a 0.10 position, MOEA/D moves it 0.0031 while Markowitz moves it 0.0103. The optimizer
learned to leave the illiquid position alone.

The three selection rules are reported side by side as the documented sensitivity analysis;
knee gives up ~4% of return for a marginally lower cost.

**RQ4 — an open problem, stated honestly.** The fuzzy GA has the *best* unstressed cost
(1.96 vs 2.22) but degrades faster than most-liquid-first (degradation ratio 10.1 vs 4.5;
mean cost 11.9 vs 9.8). Only pro-rata ever becomes infeasible (breakdown severity 1.0).

The cause is diagnosed, not hand-waved: **the fuzzy layer saturates under severe stress.**
At compound severity 1.0 every holding receives an identical `sell_priority` of 83.9, so the
priority signal carries no ordering information precisely when ordering matters most. Two
contributing factors, one fixed and one open:

1. *Fixed.* `position_liquidity_score` was linear and saturated at 20% of ADV, so a 95% ADV
   collapse floored every holding to 0. It is now log-scaled over five decades of
   participation, which makes a common ADV shock a constant offset and preserves ranking.
   Baseline spread recovered from 0 to 14.1 priority points.
2. *Open — needs a methodology decision.* The rule base maps all three of
   `(HIGH urgency, TURBULENT, {LIQUID, NORMAL, ILLIQUID})` to `VERY_HIGH`. Under high
   urgency in a turbulent market it therefore says "sell everything at maximum priority"
   regardless of liquidity, which is defensible as crisis behaviour but removes the
   discrimination RQ4 measures. Differentiating that row would likely close the gap against
   most-liquid-first. **This is a research decision, not a bug fix** — deliberately left for
   supervisor discussion rather than tuned until the proposed method wins.

## Notebooks

| Notebook | Runs where | Purpose |
|---|---|---|
| `experiments/rq_analysis.ipynb` | local | RQ1–RQ4 tables and plots; executes headlessly |
| `experiments/colab_finetune.ipynb` | **Colab (T4)** | LoRA fine-tuning + agent SFT; produces adapters |

`colab_finetune.ipynb` is the bridge between this repo and the parts the dev machine cannot
run. It clones the repo, fine-tunes TimesFM/Chronos-Bolt against the committed
`resolved_universe.yaml`, trains the residual head, produces the full RQ1 table, then
SFT-trains the tool-calling agent on the generated trajectories. It downloads **adapters
only** — a few MB against ~821MB of base weights, which is the whole reason LoRA is used
here rather than full fine-tuning.

It also asserts, before training the agent, that every trajectory called
`run_fuzzy_ga_withdrawal`. Training on even a handful of ungrounded transcripts would teach
the model that inventing a number is sometimes acceptable — the exact failure the design
exists to prevent.

## Phase 5c: the SFT dataset

`artifacts/trajectories/withdrawal_sft_scripted_800.jsonl` — 767 trajectories from 800
scenarios (4.1% rejected), 5.95 MB. This is what Colab consumes.

The file itself is **gitignored**: 6 MB of deterministically regenerable output does not
belong in version control. The generator is committed and seeded (`seed=42`), so
`uv run python run_pipeline.py --stages 6` reproduces it byte-for-byte. Generate it locally,
then upload it to the Colab session before running Part B of `colab_finetune.ipynb`.

| Property | Value |
|---|---|
| Grounded | **767 / 767** |
| Called `run_fuzzy_ga_withdrawal` | **767 / 767** |
| Market regimes | normal 421, volatility_spike 116, adv_collapse 115, compound 115 |
| Infeasible plans | 70 — kept deliberately, RQ4 needs them observable |
| Multi-day liquidations | 21 |

**A bug worth recording, because it targeted the most important cases.** `enforce_grounding`
keyed `assets_to_sell` by symbol alone. A holding is legitimately sold across several days
whenever the ADV participation cap bites — one plan sold `THIN` as
`[0.17106, 0.17106, 0.122843]` over days 0–2 — and the dict collapsed those rows to the last
one, compared day 0's fraction against day 2's, and declared a faithful plan tampered with.

Measured on 200 identical scenarios: **190/200 accepted before the fix, 200/200 after** — a
5% recovery. Small in aggregate, but not randomly distributed: it fell entirely on the
multi-day illiquid liquidations that are precisely what this component exists to handle, so
the dataset was losing exactly its hardest and most valuable examples. Now keyed on
`(symbol, execution_day)`, with a check that no step was dropped, and three regression tests
including one confirming tampering is still caught *within* a multi-day plan.

## Model policy: local first, download only when required

Nothing in this codebase starts a large download on its own. Every model-backed step checks
what is genuinely present locally and either uses it or fails with the exact command to fetch
it. This is not fastidiousness -- on a ~24 KB/s link an implicit fetch turns a pipeline run
into an overnight stall, and a silent stall is far worse than an explicit fallback.

| Step | Preferred | Local fallback | If neither |
|---|---|---|---|
| Sentiment | `ProsusAI/finbert` (~420MB) | **Ollama `gemma4-e4b`** — already installed | clear error naming both fixes |
| Forecasting | TimesFM / Chronos-Bolt (~821MB each) | **`baseline_lstm`** — trains locally, no weights | — |
| Agent | fine-tuned Llama (Colab) | **Ollama `gemma4-e4b`** | test skips |

Two distinctions the code makes carefully:

- `available_foundation_models()` means *the package imports*; `usable_foundation_models()`
  additionally requires *weights on disk*. The gap between them is hours of download, so
  anything choosing a default consults the latter. `/health` reports the latter too.
- A cache directory is not a cache hit. An interrupted fetch leaves `config.json` plus
  zero-byte `.incomplete` placeholders, which naive checks read as "present". Both
  `weights_cached()` and `finbert_weights_cached()` require a real multi-megabyte file.

`get_forecaster("hybrid")` therefore refuses with an actionable message rather than silently
starting a 10-hour fetch.

### Using the local model for sentiment

`gemma4-e4b` is prompted for the same three-way probability split FinBERT produces, decoded
with `format=json` at temperature 0 so re-running does not move the features. Verified
against calibration headlines:

| Headline | polarity | confidence |
|---|---|---|
| "Apple crushes Q4 earnings, raises guidance, $90B buyback" | **+1.000** | 1.00 |
| "Boeing halts 737 production after fatal crash; CEO resigns" | **−0.950** | 0.95 |
| "Microsoft to hold its annual shareholder meeting Thursday" | 0.000 | 0.20 |

The neutral case is the informative one: near-zero polarity *and* low confidence, so
confidence-weighted aggregation gives it almost no influence. A malformed reply yields
`(0.0, 0.0)` — deliberately distinguishable from a confident neutral.

**For the write-up:** these are not identical scorers. Record which backend produced a run's
features; mixing them inside one experiment would be a confound. FinBERT remains the
proposal-named method and is preferred whenever its weights are present.

## Known risks

- **`scikit-fuzzy` is dormant** (no release since Aug 2024, declares no numpy upper bound).
  Pure-Python, so no ABI risk, but numpy-2 runtime behaviour is unverified upstream.
  `tests/test_env.py` evaluates a real Mamdani control system to catch this. It is a
  TAF-named method, so a failure here needs a hand-written Mamdani layer, not a substitution.
- **`timesfm` on Windows** — only the `[torch]` extra installs. See the deviations table.
- **FinBERT throughput** is the pipeline's slowest step; scores are cached per headline hash.
- **This machine has an AMD Radeon 610M and no CUDA.** torch is pinned to the CPU-only index
  (see `pyproject.toml`); the default Windows PyPI wheel bundles ~2.5GB of unusable CUDA
  libraries. LoRA fine-tuning therefore belongs in Colab, with inference local on CPU.

  This is a *hazard for the code*, not only an inconvenience. Anything written to suit a
  CPU-only machine follows the repository onto a GPU one. Two bugs of exactly that shape
  have already been fixed: the fine-tuning loop had no device handling at all and blew up on
  Colab with `mat1 is on cpu, different from other tensors on cuda:0`, and `ChronosConfig`
  pinned `device="cpu"` with the comment *"this machine has no CUDA"*, which made Colab train
  on the CPU with no error at all — just twenty epochs at the wrong speed. Device is now
  resolved once per run by `forecasting/base.py::resolve_device`, and the training loop moves
  every batch to wherever the model is. Prefer auto-detection over a pin, and be suspicious
  of any default justified by this laptop.
- **Small models fail the tool-calling contract in two distinct ways**, both observed with
  gemma4-e4b and both handled in `agent/reference_agent.py`:
  1. *Looping.* It cycles the three context tools indefinitely and never calls the optimizer,
     exhausting its iteration budget. Handled by `NUDGE_AFTER_STEPS`.
  2. *Prose instead of JSON.* Having called the optimizer correctly, it narrates the plan in
     English rather than emitting the decision object, so `_parse_decision` returns None and
     the transcript is (correctly) rejected as ungrounded. Handled by a one-shot re-ask that
     hands the tool output back verbatim.

  3. *Runaway wall clock.* `ollama ps` reports gemma4-e4b running **100% on CPU with no GPU
     offload**, at roughly **150 seconds per turn**. One episode ran past 50 minutes before
     any bound existed. `run_episode` is now capped on BOTH turns and wall clock, and the
     per-request timeout matters as much as the episode budget: the deadline is only checked
     between turns, so a single blocking call would otherwise sail straight through it.

  None of these is a bug in the tools -- the grounding validator caught the first two and the
  budget caught the third, which is the point of having them. Together they are why the
  deterministic scripted policy, not the reference agent, is the default driver for bulk
  trajectory generation: 767 grounded trajectories in minutes versus roughly ten minutes per
  episode, and a fine-tuned model should learn the discipline from clean trajectories rather
  than inherit a small model's failure modes.

  The `integration` test overrides the default budget to 1200s deliberately. The 300s default
  is a SAFETY bound for anything user-facing; the test exists to validate the tool interface,
  not to enforce latency, and abandoning an episode mid-flight tells us nothing about whether
  the interface works.
- **MLflow's filesystem store is now in maintenance mode and raises**; tracking uses
  `sqlite:///artifacts/mlflow.db`.
- **yfinance signals rate-limiting exactly like a delisting** — an empty frame. Two guards:
  `_fetch_ohlcv_frame` retries with linear backoff, and `cached_fetch(cache_empty=False)`
  refuses to cache an empty OHLCV result. Without the second, a throttled symbol is cached
  as "no data" and silently vanishes from every later run. This was not hypothetical: a
  universe resolution lost SPY, XOM, BA, TLT and XLE that way before the fix. News keeps
  `cache_empty=True`, because "no headlines that week" genuinely is the answer.
- **YAML scientific notation needs an explicit sign.** YAML 1.1 parses `1.0e12` as a *string*
  and only `1.0e+12` as a float, so a missing `+` silently turns every liquidity figure into
  text that the cost model reads as untradeable. Guarded by
  `tests/test_scaffold.py::test_forex_entries_carry_notional_adv`, which asserts the type.
- **This machine's network runs at ~24 KB/s** -- measured against both HuggingFace and PyPI,
  so it is the link, not a provider. Consequences:
  - The foundation-model checkpoints (~821MB each) would take **~10 hours** to fetch. Three
    attempts left only 0-byte `.incomplete` placeholders in the HF cache.
  - The `slow` adapter tests therefore check the local HF cache first and **skip with an
    actionable message rather than hanging**. A skip there means "weights absent", not
    "adapter broken" -- the adapter code is exercised by the non-slow tests, and its API
    usage is verified against the installed classes.
  - Pre-warm with `huggingface-cli download amazon/chronos-bolt-base`, or run RQ1 in Colab.
  This is the same constraint that makes Colab the right home for fine-tuning: fast network
  and a free T4, versus a CPU-only laptop on a 24 KB/s link.

---

## Status

All eight phases are complete. **362 tests passing**, zero unimplemented functions.

| Phase | Scope | State |
|-------|-------|-------|
| 0 | Scaffold + dependency gate | complete |
| 1 | Ingestion + per-symbol window selection | complete — run on real data, 26/26 symbols |
| 2 | Technical + sentiment features | complete — sentiment runs on the local model |
| 3 | Baseline LSTM | complete — RQ1 baseline row measured |
| 4 | Foundation adapters + hybrid + registry | code complete; fusion validated locally, foundation weights pending Colab |
| 5a | MOEA/D + Markowitz + 3 Pareto rules | complete — RQ2 measured |
| 5b | Fuzzy GA withdrawal | complete — RQ3/RQ4 measured |
| 5c | Agent tools + grounding + trajectories | complete — 767 grounded trajectories |
| 6 | FastAPI + Kafka + frozen contract | complete — wired to the frontend |
| 7 | Evaluation harness + notebooks | complete — both notebooks execute |
| 8 | Docker + Kafka + MLflow + Prometheus | complete |

Remaining work is compute on other hardware, not code: LoRA fine-tuning in Colab, then
re-running RQ1 with the foundation and hybrid rows populated. See
`experiments/colab_finetune.ipynb`.

## Frontend integration

This service is consumed by the Next.js frontend in `frontend/`. Two additions support that,
both written to be copied verbatim into Components 2-4:

| File | Purpose |
|---|---|
| `service/cors.py` | The browser calls this service directly rather than through a proxy, so without CORS headers it discards the response before any handler runs. Origins come from `ALLOWED_ORIGINS`. |
| `service/auth.py` | Verifies the JWT issued by the shared platform service (`backend/Platform`) using the same `JWT_SECRET`. Verification only — this service never calls the platform on the request path, so platform downtime cannot become Component 1's. |

`require_user` is gated by `AUTH_REQUIRED`, which defaults to **off**. That keeps the
existing test suite running unauthenticated; docker-compose turns it on. A flag defaulting to
"secure" would have meant editing every one of those tests, which is how a security control
ends up disabled wholesale rather than scoped.

Ports: this service on **8000**, the platform service on **8100**, the frontend on **3000**.
Full wiring instructions for the other components are in the repository-root
[`INTEGRATION.md`](../../INTEGRATION.md).

### Interactive latency

`/portfolio/optimize` runs MOEA/D with `n_partitions=8, n_generations=100` — the same
configuration RQ2 was measured with — rather than the `MOEADConfig()` research defaults of
12 x 200. That is ~7s instead of ~30s, which matters because the TAF's claim for this
component is a *"real-time, user-facing service"*. Profiling showed the cost is pymoo's own
genetic operators, not the objective functions: objective evaluation accounts for under a
second of it. Offline analysis should pass its own `MOEADConfig` for a denser front.

## Market data and per-stock sentiment

Holdings can come from any instrument on the **Colombo Stock Exchange** (all ~282
securities), **Binance** (all spot pairs) or **Yahoo Finance**. Analysis runs only for
symbols someone holds — the *active set*.

| Endpoint | Purpose |
|---|---|
| `GET /market/search?q=` | one search across CSE, Binance and Yahoo (CSE wins ticker ties) |
| `POST /market/quote` | price, 20-day traded value, volatility, in native currency and LKR |
| `POST /analysis/track` | register a portfolio's symbols; preparation runs in the background |
| `GET /analysis/status` | `preparing` / `ready` / `partial` / `failed`, per step |
| `POST /analysis/refresh` | re-run preparation (news goes stale within hours) |
| `GET /analysis/channels` | every news/social channel, and why any cannot run |
| `GET`/`PUT /analysis/keywords/{symbol}` | the symbol's keyword profile; edits mark it reviewed |
| `GET /analysis/sentiment/{symbol}` | per-tier sentiment, attention spike, top sources |

**Preparation steps** (`data/active_set.py::PIPELINE`): `liquidity` (history + volatility
metrics, which now replace the GA's hard-coded volatility for prepared symbols), `keywords`,
`sentiment`.

**Keyword profiles** (`news/keywords.py`) are generated per symbol from the exchange's own
name, sector/macro templates in `configs/news.yaml`, and an optional local-LLM expansion
(aliases, brands, people, Sinhala/Tamil names). Reviewed profiles are never regenerated;
`configs/keyword_overrides.yaml` holds version-controlled corrections.

**Channels** (`news/channels/`):

| Channel | Needs | Notes |
|---|---|---|
| `google_news` | nothing | keyword search; Sri Lankan edition for CSE symbols |
| `gdelt` | nothing | throttled to one call per 6 s (GDELT answers faster callers with 429) |
| `rss` | nothing | 9 verified outlet feeds, region-filtered |
| `cse_announcements` | nothing | the company's own CSE disclosures (official weight) |
| `youtube` | `yt-dlp` (optional extra `social`) | titles, descriptions, engagement |
| `newsapi` | `NEWSAPI_KEY` | |
| `jina_web` | `JINA_API_KEY` | web + public LinkedIn posts |
| `twitter`, `reddit`, `instagram`, `facebook`, `linkedin` | Agent-Reach CLIs on the host | command templates in `configs/news.yaml` must be checked against the installed tools |

**Relevance** requires a word-boundary match of the profile's terms; short tickers must be
upper-case, exclusion terms veto, and for CSE symbols sector/macro items must also carry
Sri Lankan context (a term or a Sri Lankan outlet). A Laya zero-shot check is added
automatically once a Laya checkpoint is cached locally.

**Scoring**: English via FinBERT if cached, else the local Ollama model, in batches of 8
(measured 2026-10-10: gemma4-e4b scores 8 headlines in ~200 s on this CPU vs ~100 s per
headline unbatched). Sinhala/Tamil items are scored only by Laya-multilingual; without it
they count toward attention but not sentiment.

**Storage**: the aggregate block in the active set, and per-day per-tier rows in
`artifacts/sentiment/<symbol>.parquet` for the forecaster. Raw social text is never stored.

## Fundamentals and technical analysis

| Endpoint | Purpose |
|---|---|
| `GET /analysis/fundamentals/{symbol}` | latest statements, ratios, F/Z scores, sector percentiles, red flags, outlook tone (tokenomics for crypto) |
| `GET /analysis/technical/{symbol}` | trend (20/50/200-day MAs, crosses), RSI/MACD/MFI/ATR, Bollinger %B, ADX, OBV trend, 52-week drawdown, support/resistance, 63-day strength vs ASPI / SPY / BTC |

**Sources** (`fundamentals/reports.py`): CSE's per-company report archive (interim PDFs, text
layer, parsed with `pypdf`; download from `cdn.cse.lk`), yfinance quarterly statements for
US names, CoinGecko market data for crypto. Every record carries `available_from` (CSE
upload time; Yahoo period end + 45 days, flagged as estimated) and `fundamentals.pipeline.as_of`
only ever returns reports available at a given date.

**Extraction** (`fundamentals/extract.py`): statement pages found by heading (group column
first), lines split into label / note ref / current / comparative, mapped through ranked
synonym lists, units detected (`Rs '000`), brackets read as negatives. Templates: general,
bank, finance company, insurance, detected from the income statement itself. Accounting
identities are checked on every report (assets = liabilities + equity, gross profit,
profit after tax, net interest income); `confidence` < 0.5 keeps a report out of decisions.
A local-LLM read of the key fields is tried only when the parser fails. Scanned PDFs are
recorded as unusable (no OCR). `python -m evaluation.extraction_accuracy` scores the parser
against hand-checked values in `evaluation/handchecked/extraction.yaml` (2 reports so far,
both 100% field-accurate; the target is ~30 across all templates).

**Ratios** (`fundamentals/ratios.py`): margins, ROE/ROA (annualised), growth (YoY from the
comparative column, QoQ between quarters), leverage, interest cover, current ratio, CFO /
profit, dividend cover, P/E, P/B, yields, bank ratios (NIM on assets, cost/income,
impairment/loans, loans/deposits), Piotroski F (score only with ≥5 signals computable), and
Altman's Z''-EM (not for banks/finance/insurance).

Live, only the newest 2 interims per symbol are downloaded. The full archive comes from
**Colab Part D** (`experiments/colab_fundamentals_backfill.ipynb`, plus SEC EDGAR for the US
names): `python -m fundamentals.pipeline --import fundamentals_backfill.parquet`.

## Forecasting service (TimesFM + LSTM head)

Colab **Part A** (`experiments/colab_hybrid_heads.ipynb`) trains one residual head per
horizon (5 and 21 sessions) over frozen TimesFM on every CSE security with ≥200 sessions, the
top 50 Binance USDT pairs and the US universe. Inputs add the per-tier sentiment and the
point-in-time fundamentals (`features/enrichment.py`; identical column set everywhere,
zero-filled with `has_*` flags when absent). Holdout metrics (base vs hybrid pinball loss,
per asset class) travel in each head's `meta.json` into the model registry.

```bash
python -m forecasting.serving --install hybrid_heads.zip      # registers hybrid_h5 / hybrid_h21
huggingface-cli download google/timesfm-2.5-200m-pytorch      # once, ~821 MB
```

`/forecast` then serves the model and stamps its real `model_version`. Without a head or
the TimesFM weights it answers 503 with the reason. The analysis pipeline's `forecast` step
falls back to an **empirical** band (the symbol's own past h-day returns), always labelled
`method: "empirical"`; `GET /analysis/forecast/{symbol}` shows which one each band is.

Training fix made here: the head's LSTM windows are now cut from the full consecutive
history before each forecast cut (`HybridForecaster._cut_sequences`). They previously spanned
only the strided walk-forward rows, so training saw gapped sequences while serving read
consecutive days.

## Liquidation driven by forecasts, news and fundamentals

`/portfolio/withdraw` reads each holding's prepared snapshot (SQLite only; no model or
exchange call on the request path) and:

* **converts currencies**: each holding stays in its trading currency (LKR / USD / USDT); the
  plan is in `base_currency` (default LKR). The conversion spread (`FX_SPREAD_FIAT`,
  `FX_SPREAD_STABLECOIN`) is charged by the GA on foreign proceeds. Requests naming no
  base and no holding currency behave exactly as before.
* **GA terms** (`optimization/ga_withdrawal.py`): forecast return forgone (+) or loss avoided
  (−) on what is sold through the deadline; timing risk from the forecast band; FX cost;
  crystallised loss vs cost basis (weight 0.25); `risk_tolerance` now sets timing-risk
  aversion (it was accepted but unused).
* **Fuzzy signal layer** (`optimization/fuzzy_withdrawal.py`): an 18-rule Sugeno system over
  news sentiment × forecast outlook × attention spike adjusts each sell priority; the rules
  that fired are in `fuzzy_rule_trace[].signal_rules`.
* **Response** (additive fields only): `market_context` per holding, `currency`,
  `cost_breakdown`, `baseline_comparison` (the three naive baselines costed by the same
  simulator), `effective_inputs`.
* **Penalty scaling fix**: the GA's shortfall and priority penalties were absolute constants
  while costs scale with money, so at LKR amounts (~300x USD) or large positions the GA
  preferred under-raising (feasible in 6% of realistic scenarios vs 41-65% for the naive
  baselines). Both penalties are now proportional to the target; a test asserts the same
  schedule in USD and x300.
* Review fixes: `/portfolio/optimize` applies `max_weight` and uses forecast drift as
  expected return; with `use_agent` the plan returned IS the agent's tool call, so the trace
  matches the numbers.

## Laya decision model (two stages)

`decision/`: Stage 1 asks technical, fundamental and news questions per holding (background
preparation, step `laya_analysis`); Stage 2 asks per holding *sell now / spread / defer /
exclude* and *downside risk*, and per portfolio *urgency* and *risk tolerance* (request
path, one batched pass). Answers with confidence ≥ 0.6 adjust fuzzy priorities, cap an
excluded holding at zero (only if the rest can cover), and replace urgency / risk tolerance;
the GA still computes every amount. Below threshold, or without Laya, the fuzzy rules decide
and `market_context[symbol].decision` says why.

Colab **Part C** (`experiments/colab_laya_decision.ipynb`) builds outcome-labelled,
point-in-time training data (`decision/labels.py`), fine-tunes with the encoder frozen,
evaluates zero-shot vs fine-tuned on the dates after the cutoff, and exports a head-only zip:

```bash
uv sync --extra decision
huggingface-cli download convaiinnovations/laya               # once, ~842 MB
python -m decision.checkpoint --install laya_head.zip         # registers laya_decision
```

Verified locally: `laya` 0.4.1 installs and imports on Windows / Python 3.12 / CPU torch, and
every training row passes laya's own `to_internal` / `target_from_gold`. Not verified
locally: inference with real weights (not downloaded on this link).

## Evaluation (Phase 6)

| Script | Answers |
|---|---|
| `python -m experiments.rq_liquidation --bars bars.parquet` | RQ3/RQ5: naive baselines vs fuzzy GA vs + forecast vs + oracle drift vs Laya, scored on **realised** implementation shortfall, stratified by liquidity |
| `python -m evaluation.extraction_accuracy` | statement-extraction field accuracy and template accuracy |
| `python -m evaluation.sentiment_precision export … / score` | precision of the keyword + relevance filter on a hand-labelled sample |
| Part A `meta.json` metrics | RQ1 base vs hybrid pinball per asset class |
| Part C `evaluation.csv` | RQ5 zero-shot vs fine-tuned Laya per question |

**Indicative local result (2026-10-10, not the RQ3 result).** 80 scenarios over the 5 symbols
cached locally (JKH, AAPL, SPY, BTC, SOL), empirical bands only: after the penalty fix the GA
is feasible as often as the best baselines (65%) and within ~1-2 bps of them on realised
shortfall. On terminal wealth vs holding, the oracle-drift GA beats the plain GA by ~14 bps
while the empirical band adds nothing -- the forecast pathway works when the forecast is
right, so RQ3 hinges on the trained hybrid. Run the real ablation on Colab over Part A's
`bars.parquet` with the head installed.

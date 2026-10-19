"""Red flags and outlook tone from a report's text.

FLAGS (regex, always): going-concern doubt, qualified/adverse/disclaimer audit opinion,
emphasis of matter, and contingent liabilities relative to equity. Statutory wording for
these is formulaic (SLAuS 570 / ISA 570 "material uncertainty ... going concern"), so a
pattern finds them reliably; a pattern that merely mentions "going concern basis" (the
standard accounting-policy note) is NOT a flag.

OUTLOOK TONE (typed question): the management commentary (chairperson's message / review) is
answered as positive / neutral / negative by, in order of preference,
  1. Laya-multilingual (typed `choice`), when its checkpoint is cached;
  2. the local Ollama model (JSON-constrained);
  3. a small finance word list (Loughran-McDonald style) -- recorded as `method: lexicon`.
The method used is always reported; a lexicon answer is never presented as a model's.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

COMMENTARY_CHARS = 3000

_GOING_CONCERN = re.compile(
    r"material uncertaint\w*[^.]{0,200}going concern|significant doubt[^.]{0,120}going concern|"
    r"going concern[^.]{0,120}(material uncertaint|significant doubt)", re.I | re.S)
_QUALIFIED = re.compile(r"\b(qualified opinion|adverse opinion|disclaimer of opinion|except for the (effects?|possible effects?))", re.I)
_EMPHASIS = re.compile(r"emphasis of matter", re.I)
_UNAUDITED = re.compile(r"(figures|statements) (are|have) not (been )?audited|unaudited", re.I)
_COMMENTARY = re.compile(r"CHAIRPERSON|CHAIRMAN|MESSAGE|REVIEW OF OPERATIONS|MANAGEMENT DISCUSSION|OUTLOOK|COMMENTARY", re.I)
_STATEMENT = re.compile(r"STATEMENT OF (FINANCIAL POSITION|CASH FLOWS?|PROFIT|COMPREHENSIVE)|INCOME STATEMENT", re.I)

_POSITIVE = {"growth", "increased", "improved", "improvement", "strong", "record", "momentum",
             "recovery", "exceeding", "expansion", "positive", "resilient", "higher", "gain"}
_NEGATIVE = {"decline", "declined", "decrease", "decreased", "loss", "losses", "weak", "weakened",
             "challenging", "disrupted", "impairment", "adverse", "uncertainty", "lower", "pressure",
             "downturn", "deteriorat"}

_TONE_QUESTION = {
    "outlook": {
        "type": "choice",
        "instructions": "Read this company management commentary. For a shareholder, the outlook it describes is:",
        "criteria": {"positive": "improving prospects", "neutral": "mixed or unchanged",
                     "negative": "worsening prospects"},
    }
}


def commentary(pages: dict[int, str]) -> str:
    """Management commentary: pages that look like a message/review and are not statements."""
    chunks = [text for _, text in sorted(pages.items())
              if _COMMENTARY.search(text[:400]) and not _STATEMENT.search(text)]
    return re.sub(r"\s+", " ", " ".join(chunks)).strip()[:COMMENTARY_CHARS]


def flags(pages: dict[int, str], statement: dict[str, Any] | None = None) -> dict[str, Any]:
    text = "\n".join(pages[i] for i in sorted(pages))
    out: dict[str, Any] = {
        "going_concern_doubt": bool(_GOING_CONCERN.search(text)),
        "modified_audit_opinion": bool(_QUALIFIED.search(text)),
        "emphasis_of_matter": bool(_EMPHASIS.search(text)),
        "unaudited": bool(_UNAUDITED.search(text)),
    }
    contingent = None
    match = re.search(r"contingent liabilities(?: and commitments)?\s+([\d,]{5,})", text, re.I)
    equity = ((statement or {}).get("current") or {}).get("total_equity")
    # Banks' contingents are guarantees and credit lines -- their business, not a red flag.
    financial = (statement or {}).get("template") in ("bank", "finance", "insurance")
    if match and equity and not financial:
        unit = (statement or {}).get("unit") or 1.0
        contingent = float(match.group(1).replace(",", "")) * unit / equity
    out["contingent_to_equity"] = round(contingent, 3) if contingent is not None else None
    return out


def _lexicon_tone(text: str) -> dict[str, Any]:
    words = re.findall(r"[a-z]+", text.lower())
    positive = sum(1 for w in words if w in _POSITIVE)
    negative = sum(1 for w in words if any(w.startswith(n) for n in _NEGATIVE))
    total = positive + negative
    if total == 0:
        return {"tone": "neutral", "confidence": 0.0, "method": "lexicon"}
    balance = (positive - negative) / total
    tone = "positive" if balance > 0.2 else "negative" if balance < -0.2 else "neutral"
    return {"tone": tone, "confidence": round(min(1.0, abs(balance)), 3), "method": "lexicon"}


def outlook_tone(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    from news.scoring import laya_router

    router = laya_router()
    if router is not None:
        try:
            answer = router.predict(text, _TONE_QUESTION, model="multilingual")["answers"]["outlook"]
            choice = answer.get("choice")
            if choice in ("positive", "neutral", "negative"):
                probabilities = answer.get("probabilities") or answer.get("distribution") or {}
                return {"tone": choice, "confidence": float(probabilities.get(choice, 0.5)) if probabilities else 0.5,
                        "method": "laya"}
        except Exception as exc:  # noqa: BLE001
            logger.info("laya outlook question failed: %s", exc)

    from news.llm import chat_json

    try:
        payload = chat_json(
            "You classify the outlook in company management commentary. Reply with JSON "
            '{"tone": "positive"|"neutral"|"negative", "confidence": 0..1}.',
            text, timeout=180,
        )
        if payload.get("tone") in ("positive", "neutral", "negative"):
            return {"tone": payload["tone"], "confidence": float(payload.get("confidence", 0.5)), "method": "ollama"}
    except Exception as exc:  # noqa: BLE001 - Ollama down is a normal state
        logger.info("ollama outlook question unavailable: %s", exc)
    return _lexicon_tone(text)

"""Helpers for the Colab SFT of the tool-calling agent (colab_finetune.ipynb, Part B).

Kept out of the notebook so they can be tested without a GPU. Each exists because the
notebook got the thing wrong once:

  * a T4 has no native bf16, yet the cell asked for it
  * `load_in_4bit=True` passed straight to `from_pretrained` is deprecated; a
    BitsAndBytesConfig is the supported route
  * `torch_dtype=` was renamed `dtype=` in transformers 4.56
  * a chat template can silently drop `tool` turns, leaving the model to learn tool calls
    with no tool results -- the exact thing the grounding constraint exists to prevent
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Precision:
    bf16: bool
    fp16: bool
    compute_dtype: str          # "bfloat16" | "float16" | "float32"


def choose_precision(capability: tuple[int, int] | None) -> Precision:
    """Training precision for a CUDA compute capability (None = no GPU).

    bf16 needs Ampere (8.0) or newer. Turing parts such as the Colab T4 (7.5) get fp16.
    """
    if capability is None:
        return Precision(bf16=False, fp16=False, compute_dtype="float32")
    if capability[0] >= 8:
        return Precision(bf16=True, fp16=False, compute_dtype="bfloat16")
    return Precision(bf16=False, fp16=True, compute_dtype="float16")


def four_bit_params(compute_dtype: str) -> dict[str, Any]:
    """Keyword arguments for transformers.BitsAndBytesConfig (QLoRA's NF4 setup).

    `compute_dtype` is a name, not a torch dtype, so this stays importable without torch;
    the notebook maps it with getattr(torch, name).
    """
    return {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": compute_dtype,
    }


def dtype_kwarg(transformers_version: str) -> str:
    """"dtype" on transformers >= 4.56, "torch_dtype" before it."""
    parts = transformers_version.split(".")
    version = (int(parts[0]), int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0)
    return "dtype" if version >= (4, 56) else "torch_dtype"


def render_messages(record: dict) -> list[dict[str, str]]:
    """A trajectory's messages in the shape a chat template takes.

    An assistant turn that called tools has its calls serialised into `content`, because
    the templates do not agree on a tool-call schema and the model must learn the format the
    runtime will parse. Every content is a string; tool results keep their `tool` role.
    """
    messages = []
    for message in record["messages"]:
        content = message.get("content") or ""
        if message.get("tool_calls"):
            content = json.dumps(message["tool_calls"])
        messages.append({"role": message["role"], "content": content})
    return messages


def assert_template_keeps_tool_turns(render: Callable[[list[dict[str, str]]], str], record: dict) -> None:
    """Raise unless a rendered trajectory still contains every tool result.

    `render` is the tokenizer's chat template applied to the messages. Some templates
    ignore or reject the `tool` role; the failure is silent, and training on text with the
    tool results missing teaches the model to emit plans it never saw grounded.
    """
    messages = render_messages(record)
    tool_turns = [m for m in messages if m["role"] == "tool" and m["content"]]
    if not tool_turns:
        return

    # Differential, not a substring search: the final assistant answer routinely repeats the
    # start of the tool result verbatim, so "the tool text appears in the output" is true even
    # when the template threw the tool turn away. A template that keeps the turns renders
    # longer by at least the tool content; one that drops them renders the same.
    without = [m for m in messages if m["role"] != "tool"]
    grew_by = len(render(messages)) - len(render(without))
    expected = sum(len(m["content"]) for m in tool_turns)
    if grew_by < 0.9 * expected:
        raise ValueError(
            f"the chat template dropped the tool turn(s): rendering them added {grew_by} characters "
            f"against {expected} of tool output. Pick a base model whose template supports the "
            "tool role, or render tool results into the user turn before training."
        )


def library_versions(names: list[str]) -> dict[str, str | None]:
    """Installed version of each distribution, or None when it is not installed.

    Written into adapters_meta.json and printed before the SFT, so a run on Colab -- where the
    ML stack floats with the runtime image -- can be reproduced or diagnosed afterwards without
    guessing which transformers/trl pairing it used.
    """
    from importlib.metadata import PackageNotFoundError, version

    found: dict[str, str | None] = {}
    for name in names:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            found[name] = None
    return found


def repo_access_problem(repo_id: str) -> str | None:
    """Why this account cannot download `repo_id`, or None when it can.

    Fetches only config.json, a few hundred bytes, so the check costs a second rather than the
    multi-gigabyte download it protects. Gated repositories (Meta's Llama) answer 403 until the
    account has accepted the licence AND been approved, which can take a while; finding that out
    after the notebook has installed everything and loaded a dataset is what this avoids.
    """
    import huggingface_hub
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    try:
        huggingface_hub.hf_hub_download(repo_id, "config.json")
    except GatedRepoError:
        return (f"gated: accept the licence at https://huggingface.co/{repo_id}, wait for approval, "
                "and log in with that account")
    except RepositoryNotFoundError:
        return "not found, or private to another account"
    except Exception as exc:  # noqa: BLE001 - any other failure is still a reason not to start training on it
        return f"{type(exc).__name__}: {str(exc)[:120]}"
    return None


def resolve_base_model(
    candidates: list[str],
    check: Callable[[str], str | None] = repo_access_problem,
) -> tuple[str, dict[str, str]]:
    """The first candidate this account can download, plus {skipped candidate: reason}.

    Order is preference: put the model you want first (Llama, which the TAF names) and an ungated
    one after it (Qwen2.5, which the existing agent adapter was trained on). Raises, listing every
    reason, when none is reachable.
    """
    skipped: dict[str, str] = {}
    for repo_id in candidates:
        problem = check(repo_id)
        if problem is None:
            return repo_id, skipped
        skipped[repo_id] = problem
    detail = "; ".join(f"{repo}: {reason}" for repo, reason in skipped.items())
    raise RuntimeError(f"no SFT base model is reachable from this account ({detail})")

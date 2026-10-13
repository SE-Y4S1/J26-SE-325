"""Bringing a fine-tuned Laya home over a slow link.

A saved Laya checkpoint is the whole model (~420 MB fp16). Colab Part C fine-tunes with the
encoder FROZEN, so only the decision head and the calibration change; `export_head` packs just
those (a few MB) plus the run's config (temperatures, budgets). Locally, `install_head`
rebuilds the full checkpoint by laying that head over the base checkpoint already in the
HuggingFace cache -- the base itself must be downloaded once (`huggingface-cli download
convaiinnovations/laya`), exactly like TimesFM's weights.

    python -m decision.checkpoint --install laya_head.zip
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import tempfile
import zipfile
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

ENCODER_PREFIX = "encoder."
HEAD_FILE = "head.safetensors"


def export_head(finetuned_dir: Path, out_zip: Path, *, metrics: dict | None = None,
                train_window: dict | None = None) -> Path:
    """Zip the non-encoder weights + config of a fine-tuned checkpoint."""
    from safetensors.torch import load_file, save_file

    finetuned_dir = Path(finetuned_dir)
    state = load_file(str(finetuned_dir / "model.safetensors"))
    head = {k: v for k, v in state.items() if not k.startswith(ENCODER_PREFIX)}
    with tempfile.TemporaryDirectory() as tmp:
        save_file(head, os.path.join(tmp, HEAD_FILE))
        shutil.copy(finetuned_dir / "rl_agent_config.json", os.path.join(tmp, "rl_agent_config.json"))
        Path(tmp, "meta.json").write_text(json.dumps({
            "base": "convaiinnovations/laya", "head_keys": len(head), "metrics": metrics or {},
            "train_window": train_window or {},
        }, indent=1), encoding="utf-8")
        out_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in (HEAD_FILE, "rl_agent_config.json", "meta.json"):
                archive.write(os.path.join(tmp, name), name)
    return out_zip


def _base_dir() -> Path:
    """The cached base checkpoint directory. Never downloads."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from laya.train import resolve_checkpoint_dir

    from decision.engine import BASE_REPO

    try:
        return Path(resolve_checkpoint_dir(BASE_REPO))
    except Exception as exc:  # noqa: BLE001
        raise FileNotFoundError(
            f"base Laya checkpoint is not cached ({exc}). Download it once: huggingface-cli download {BASE_REPO}"
        ) from exc


def install_head(source: Path, *, base_dir: Path | None = None, activate: bool = True) -> str:
    """Merge an exported head over the base checkpoint, register it, return its version."""
    from safetensors.torch import load_file, save_file

    from decision.engine import CHECKPOINT_ROOT, MODEL_NAME, reset
    from forecasting.model_registry import compute_content_hash, register

    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(source) as archive:
            archive.extractall(tmp)
        head_dir = Path(tmp)
        digest = compute_content_hash(head_dir)
        dest = CHECKPOINT_ROOT / digest[:12]
        if not (dest / "model.safetensors").exists():
            base = Path(base_dir) if base_dir else _base_dir()
            merged = load_file(str(base / "model.safetensors"))
            head = load_file(str(head_dir / HEAD_FILE))
            unknown = set(head) - set(merged)
            if unknown:
                raise ValueError(f"head has keys the base does not: {sorted(unknown)[:5]}")
            merged.update({k: v.to(merged[k].dtype) for k, v in head.items()})
            dest.mkdir(parents=True, exist_ok=True)
            save_file(merged, str(dest / "model.safetensors"))
            for sub in ("encoder", "tokenizer"):
                if (base / sub).exists():
                    shutil.copytree(base / sub, dest / sub, dirs_exist_ok=True)
            shutil.copy(head_dir / "rl_agent_config.json", dest / "rl_agent_config.json")
        meta = json.loads((head_dir / "meta.json").read_text(encoding="utf-8"))
        window = meta.get("train_window") or {}
        record = register(
            MODEL_NAME, head_dir,
            train_start=date.fromisoformat(window.get("start", "1970-01-01")),
            train_end=date.fromisoformat(window.get("end", "1970-01-01")),
            metrics={k: float(v) for k, v in (meta.get("metrics") or {}).items()
                     if isinstance(v, (int, float))},
            activate=activate,
        )
    reset()   # the next request loads the new head
    return record.model_version


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install a fine-tuned Laya decision head.")
    parser.add_argument("--install", type=Path, required=True, help="laya_head.zip from Colab Part C")
    parser.add_argument("--base-dir", type=Path, default=None, help="local base checkpoint (default: HF cache)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    print("registered", install_head(args.install, base_dir=args.base_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Logged-in social channels via the CLIs Agent-Reach installs (twitter-cli, OpenCLI).

Agent-Reach (github.com/Panniantong/Agent-Reach) has no Python API: it installs and health-
checks upstream command-line tools. So each channel here is a command template from
configs/news.yaml, run as a subprocess, with its JSON output mapped onto RawItem through a
list of candidate field names (each tool names things differently).

STATUS REPORTING
  "not installed"   the binary is not on PATH -- run Agent-Reach's installer on this host.
  "not configured"  a required env var (e.g. TWITTER_AUTH_TOKEN) is missing.
These are normal states, not errors: the rest of the pipeline runs without these channels.

WHERE THIS RUNS
OpenCLI drives a logged-in desktop Chrome, which does not exist inside the service
container. Run collection on the host (`python -m news.pipeline --all`), which writes to the
same artifacts/ directory the container mounts.

ACCOUNT SAFETY
Agent-Reach warns that cookie-based automation can get accounts banned. Use a dedicated
account, never a personal one.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from datetime import datetime, timezone
from typing import Any

from news.channels.base import Channel, ChannelStatus, search_terms
from news.models import KeywordProfile, RawItem, SourceKind

logger = logging.getLogger(__name__)

COMMAND_TIMEOUT_SECONDS = 90


class AgentReachChannel(Channel):
    requires_login = True

    def __init__(self, name: str, spec: dict[str, Any], fields: dict[str, list[str]]) -> None:
        self.name = name
        self.kind = SourceKind(spec.get("kind", "social"))
        self.binary = spec["binary"]
        self.command = list(spec["command"])
        self.requires_env = list(spec.get("requires_env") or [])
        self.fields = fields

    def status(self) -> ChannelStatus:
        if shutil.which(self.binary) is None:
            return self._status(False, f"not installed ({self.binary} not on PATH; run Agent-Reach's installer)")
        missing = [var for var in self.requires_env if not os.getenv(var)]
        if missing:
            return self._status(False, f"not configured (set {', '.join(missing)})")
        return self._status(True)

    def fetch(self, profile: KeywordProfile, since: datetime, *, limit: int = 20) -> list[RawItem]:
        items: list[RawItem] = []
        # Two most specific names, searched separately: CLI search syntax rarely supports OR.
        for term in search_terms(profile, max_terms=2):
            args = [part.format(query=term, limit=limit) for part in self.command]
            try:
                completed = subprocess.run(
                    args, capture_output=True, text=True, encoding="utf-8",
                    timeout=COMMAND_TIMEOUT_SECONDS, check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                logger.warning("%s: %s failed: %s", self.name, args[0], exc)
                continue
            if completed.returncode != 0:
                logger.warning("%s exited %d: %s", self.name, completed.returncode, completed.stderr[:300])
                continue
            items.extend(self.parse(completed.stdout))
        return [item for item in items if item.published_at >= since]

    # ----------------------------------------------------------------------------------

    def parse(self, stdout: str) -> list[RawItem]:
        """JSON array, an object wrapping one, or JSON lines."""
        records = _decode_records(stdout)
        items = []
        for record in records:
            item = self._to_item(record)
            if item is not None:
                items.append(item)
        return items

    def _field(self, record: dict[str, Any], key: str) -> Any:
        for name in self.fields.get(key, [key]):
            value = record.get(name)
            if value not in (None, ""):
                return value.get("name") or value.get("username") if isinstance(value, dict) else value
        return None

    def _to_item(self, record: dict[str, Any]) -> RawItem | None:
        text = self._field(record, "text")
        title = self._field(record, "title") or (str(text)[:140] if text else None)
        published = _parse_time(self._field(record, "published_at"))
        if not title or published is None:
            return None
        engagement = {}
        for key in ("likes", "shares", "comments"):
            value = self._field(record, key)
            try:
                if value is not None:
                    engagement[key] = float(value)
            except (TypeError, ValueError):
                pass
        return RawItem(
            channel=self.name, kind=self.kind, title=str(title), text=str(text) if text else None,
            url=self._field(record, "url"), published_at=published,
            author=str(self._field(record, "author")) if self._field(record, "author") else None,
            engagement=engagement,
        )


def _decode_records(stdout: str) -> list[dict[str, Any]]:
    stdout = stdout.strip()
    if not stdout:
        return []
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        records = []
        for line in stdout.splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(value)
        return records
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        for key in ("data", "results", "items", "tweets", "posts"):
            if isinstance(payload.get(key), list):
                return [r for r in payload[key] if isinstance(r, dict)]
        return [payload]
    return []


def _parse_time(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        seconds = value / 1000 if value > 1e11 else value      # ms vs s epoch
        return datetime.fromtimestamp(seconds, tz=timezone.utc).replace(tzinfo=None)
    text = str(value).strip()
    try:
        stamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        import email.utils

        try:
            stamp = email.utils.parsedate_to_datetime(text)   # Twitter's 'Wed Oct 08 ...' form
        except (TypeError, ValueError):
            try:
                stamp = datetime.strptime(text, "%a %b %d %H:%M:%S %z %Y")
            except ValueError:
                return None
    return stamp.astimezone(timezone.utc).replace(tzinfo=None) if stamp.tzinfo else stamp

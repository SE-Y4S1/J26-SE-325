"""
Request timing and performance instrumentation for Component 4.
Measures total duration, LLM stages (tool selection, synthesis), tool executions,
and upstream platform API calls with correlated request IDs and structured logging.
"""

import contextvars
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional


logger = logging.getLogger("finagent.timing")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [finagent.timing] %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class RequestTimingContext:
    def __init__(self, request_id: Optional[str] = None):
        self.request_id: str = request_id or f"req-{uuid.uuid4().hex[:8]}"
        self.start_perf: float = time.perf_counter()
        self.end_perf: Optional[float] = None

        # Stage timings in milliseconds
        self.initial_tool_selection_ms: Optional[float] = None
        self.final_synthesis_ms: Optional[float] = None
        self.general_inference_ms: Optional[float] = None
        self.total_ollama_inference_ms: float = 0.0

        # Detailed call records
        self.ollama_calls: List[Dict[str, Any]] = []
        self.tool_calls: List[Dict[str, Any]] = []
        self.http_calls: List[Dict[str, Any]] = []

    def record_ollama(
        self,
        stage: str,
        duration_ms: float,
        success: bool = True,
        error: Optional[str] = None,
    ):
        self.total_ollama_inference_ms += duration_ms
        if stage == "initial_tool_selection":
            self.initial_tool_selection_ms = duration_ms
        elif stage == "final_synthesis":
            self.final_synthesis_ms = duration_ms
        elif stage == "general_inference":
            self.general_inference_ms = duration_ms

        record = {
            "stage": stage,
            "duration_ms": round(duration_ms, 2),
            "success": success,
        }
        if error:
            record["error"] = str(error)
        self.ollama_calls.append(record)

    def record_tool(
        self,
        tool_name: str,
        duration_ms: float,
        success: bool = True,
        retries: int = 0,
        error: Optional[str] = None,
    ):
        # Count prior invocations of this tool in the same request to identify repeated calls
        prior_calls = sum(1 for c in self.tool_calls if c["tool_name"] == tool_name)
        record = {
            "tool_name": tool_name,
            "duration_ms": round(duration_ms, 2),
            "success": success,
            "retries": retries,
            "call_index": prior_calls + 1,
            "is_repeated": prior_calls > 0,
        }
        if error:
            record["error"] = str(error)
        self.tool_calls.append(record)

    def record_http(
        self,
        endpoint: str,
        duration_ms: float,
        status_code: Optional[int],
        retries: int = 0,
        is_timeout: bool = False,
        error: Optional[str] = None,
    ):
        record = {
            "endpoint": endpoint,
            "duration_ms": round(duration_ms, 2),
            "status_code": status_code,
            "retries": retries,
            "is_timeout": is_timeout,
            "success": (status_code is not None and 200 <= status_code < 300),
        }
        if error:
            record["error"] = str(error)
        self.http_calls.append(record)

    def finish(self) -> Dict[str, Any]:
        if self.end_perf is None:
            self.end_perf = time.perf_counter()
        total_duration_ms = (self.end_perf - self.start_perf) * 1000

        # Aggregate tool durations
        portfolio_tool_ms = sum(
            c["duration_ms"]
            for c in self.tool_calls
            if c["tool_name"] == "get_portfolio_analysis"
        )
        total_tool_duration_ms = sum(c["duration_ms"] for c in self.tool_calls)

        # Aggregate HTTP durations
        platform_api_ms = sum(c["duration_ms"] for c in self.http_calls)

        # Count repeated calls and retries
        repeated_tool_calls = sum(1 for c in self.tool_calls if c.get("is_repeated"))
        total_retries = sum(c.get("retries", 0) for c in self.tool_calls) + sum(
            c.get("retries", 0) for c in self.http_calls
        )
        timeouts = sum(1 for c in self.http_calls if c.get("is_timeout"))

        summary = {
            "request_id": self.request_id,
            "total_duration_ms": round(total_duration_ms, 2),
            "initial_tool_selection_ms": (
                round(self.initial_tool_selection_ms, 2)
                if self.initial_tool_selection_ms is not None
                else None
            ),
            "final_synthesis_ms": (
                round(self.final_synthesis_ms, 2)
                if self.final_synthesis_ms is not None
                else None
            ),
            "general_inference_ms": (
                round(self.general_inference_ms, 2)
                if self.general_inference_ms is not None
                else None
            ),
            "total_ollama_inference_ms": round(self.total_ollama_inference_ms, 2),
            "portfolio_tool_duration_ms": round(portfolio_tool_ms, 2),
            "total_tool_duration_ms": round(total_tool_duration_ms, 2),
            "platform_api_duration_ms": round(platform_api_ms, 2),
            "tool_call_count": len(self.tool_calls),
            "repeated_tool_calls": repeated_tool_calls,
            "retries_count": total_retries,
            "timeouts_count": timeouts,
            "tool_calls": self.tool_calls,
            "http_calls": self.http_calls,
            "ollama_calls": self.ollama_calls,
        }
        return summary


_current_timing: contextvars.ContextVar[Optional[RequestTimingContext]] = (
    contextvars.ContextVar("_current_timing", default=None)
)


def get_current_timing() -> Optional[RequestTimingContext]:
    return _current_timing.get()


def start_request_timing(
    request_id: Optional[str] = None,
) -> tuple[RequestTimingContext, Any]:
    ctx = RequestTimingContext(request_id=request_id)
    token = _current_timing.set(ctx)
    return ctx, token


def end_request_timing(token: Any) -> Optional[Dict[str, Any]]:
    ctx = _current_timing.get()
    summary = None
    if ctx is not None:
        summary = ctx.finish()
        # Emit structured log
        try:
            logger.info("REQUEST_TIMING %s", json.dumps(summary))
        except Exception:
            pass
    _current_timing.reset(token)
    return summary

"""Provider-agnostic evaluation layer.

This is the missing piece between the research methodology and the provider
adapters. DOC1 specifies the monitoring contract -- M3 ("every node execution
emits a span with resource, tokens, cost, latency, validation result") and the
Experience Record schema (5.4, `metrics: {success, cost_usd, latency_ms, ...}`)
-- but nothing in the kernel implemented it. This module adds that layer in the
shape DOC1 asks for, without touching the scheduler, the registry routing, or
the Failure Manager.

Design rules (each maps to a DOC1 requirement):

  * Provider-agnostic core. `evaluation.py` imports no provider module. A
    provider feeds this layer by returning either a plain `str` (the
    `capabilities/*.run` convention) or a structured result object carrying
    attributes like `.text`, `.model`, `.provider`, `.usage` (e.g. the HF
    adapter's `HFResult`). Normalization reads those attributes generically, so
    adding Ollama / HF / any other provider never changes the benchmark code.

  * Capture only what is measurable. Latency is always measured here (the one
    metric every execution exposes). Token counts are captured only when the
    provider's result actually carries them (HF exposes usage; the Groq-backed
    `capabilities/*.run` functions return plain strings, so their token counts
    are honestly `None` rather than estimated).

  * No fabricated cost. DOC1's schema lists `cost_usd`, but no current provider
    exposes per-call cost through this layer, so `cost_usd` is always `None`.
    Estimating it from token counts would invent a metric; it stays unmeasured
    until a provider actually returns it.

  * No provider-specific scoring. `success` is execution-level (did the call
    return without raising), `error_category` is a taxonomy shared across
    providers, and the optional `evaluation` field only reports a caller-supplied
    expected-output comparison. Quality scoring remains where the methodology
    puts it: the Learning Manager's validators (DOC1 M9), not this layer.

Structure (per the requested diagram)::

    Evaluation Framework  (benchmark() / run_case_on())
        |
        v
    Normalized Result     (ExecutionMetrics)
        |
        v
    Evaluation Metrics    (summarize())
"""

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from providers.errors import (
    ProviderAuthenticationError,
    ProviderBadRequestError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)

# Error taxonomy shared across providers. Categories are factual (drawn from
# the typed ProviderError boundary or the exception's own type name), never a
# provider-specific scoring signal.
_ERROR_CATEGORY_MAP = (
    (ProviderAuthenticationError, "auth"),
    (ProviderRateLimitError, "rate_limit"),
    (ProviderTimeoutError, "timeout"),
    (ProviderBadRequestError, "bad_request"),
    (ProviderUnavailableError, "unavailable"),
)


def classify_error(exc: Exception) -> str:
    """Map an exception to a stable category, else its type name."""
    for exc_type, category in _ERROR_CATEGORY_MAP:
        if isinstance(exc, exc_type):
            return category
    return type(exc).__name__


# ---------------------------------------------------------------------------
# Normalized Result (the provider-agnostic per-execution record)
# ---------------------------------------------------------------------------


@dataclass
class ExecutionMetrics:
    """One normalized inference execution, as any provider represents it."""

    resource_id: str
    success: bool
    timestamp: str  # ISO 8601 (UTC)
    # Attribution. Falls back to whatever the result object itself carries when
    # the experiment does not supply it explicitly.
    provider: Optional[str] = None
    model: Optional[str] = None
    task: Optional[str] = None
    # Measured here for every execution.
    latency_ms: Optional[int] = None
    # Only when the provider exposes them; None otherwise. Never estimated.
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    # Only when the call raised; None on success.
    error_category: Optional[str] = None
    # The normalized output text (measured, not scored).
    text: Optional[str] = None
    # Optional caller-supplied expected-output check. Absent unless a reference
    # answer was given -- the layer never invents one.
    evaluation: Optional[Dict[str, Any]] = None
    # Always None today: no provider exposes per-call cost through this layer.
    # Never estimated (would fabricate a metric).
    cost_usd: Optional[float] = None


def _extract_usage(usage: Any) -> tuple:
    """Pull token counts from a usage dict or any usage-like object."""
    if not usage:
        return None, None, None
    if isinstance(usage, dict):
        return (
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            usage.get("total_tokens"),
        )
    return (
        getattr(usage, "prompt_tokens", None),
        getattr(usage, "completion_tokens", None),
        getattr(usage, "total_tokens", None),
    )


def _extract_text(raw: Any) -> Optional[str]:
    if isinstance(raw, str):
        return raw
    if raw is None:
        return None
    return getattr(raw, "text", None)


def normalize_result(
    *,
    raw: Any,
    resource_id: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    task: Optional[str] = None,
    latency_ms: Optional[int] = None,
    error: Optional[Exception] = None,
    expected: Optional[str] = None,
) -> ExecutionMetrics:
    """Turn any provider result (str or structured object) into normalized metrics.

    Attribution precedence: explicit `provider`/`model` arguments first, then
    whatever the result object itself carries. Token counts come from the
    result's `usage` attribute when present.
    """
    text = _extract_text(raw)
    prompt_tokens, completion_tokens, total_tokens = _extract_usage(
        getattr(raw, "usage", None) if not isinstance(raw, str) else None
    )
    if model is None and not isinstance(raw, str):
        model = getattr(raw, "model", None)
    if provider is None and not isinstance(raw, str):
        provider = getattr(raw, "provider", None)

    success = error is None
    evaluation = None
    if success and expected is not None and text is not None:
        evaluation = {"expected": expected, "passed": expected in text}

    return ExecutionMetrics(
        resource_id=resource_id,
        provider=provider,
        model=model,
        task=task,
        success=success,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        error_category=None if error is None else classify_error(error),
        timestamp=datetime.now(timezone.utc).isoformat(),
        text=text,
        evaluation=evaluation,
    )


# ---------------------------------------------------------------------------
# Evaluation Framework -- the one benchmark code path for every provider
# ---------------------------------------------------------------------------


def run_case(
    *,
    invoke: Callable,
    resource_id: str,
    input_text: str,
    instruction: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    task: Optional[str] = None,
    expected: Optional[str] = None,
) -> ExecutionMetrics:
    """Execute one case through `invoke(input_text, instruction=...)` and
    normalize the result. `invoke` may be any provider's run_fn (plain-str
    contract) or a result-returning wrapper (e.g. the HF adapter's
    `run_result`); the code here is identical either way.
    """
    started = time.monotonic()
    raw = None
    error = None
    try:
        raw = invoke(input_text, instruction=instruction)
    except Exception as exc:  # noqa: BLE001 -- classification is the point
        error = exc
    latency_ms = int((time.monotonic() - started) * 1000)

    return normalize_result(
        raw=raw,
        resource_id=resource_id,
        provider=provider,
        model=model,
        task=task,
        latency_ms=latency_ms,
        error=error,
        expected=expected,
    )


@dataclass
class EvalCase:
    """One benchmark case. `invoke` defaults to `registry.run_fn(resource_id)`.

    Provider-specific invocation setup (e.g. pointing HF at a mocked or live
    client) is case DATA, not benchmark logic -- the loop below never changes
    when a provider is added.
    """

    resource_id: str
    input_text: str
    instruction: Optional[str] = None
    task: Optional[str] = None
    expected: Optional[str] = None
    invoke: Optional[Callable] = None


def run_case_on(registry, case: EvalCase) -> ExecutionMetrics:
    """Benchmark one case against a registry, attributing provider/model from
    the resource's manifest metadata (trace data, never decision logic)."""
    manifest = registry.describe(case.resource_id)
    invoke = case.invoke or registry.run_fn(case.resource_id)
    return run_case(
        invoke=invoke,
        resource_id=case.resource_id,
        input_text=case.input_text,
        instruction=case.instruction,
        provider=manifest.metadata.get("provider"),
        model=manifest.metadata.get("model"),
        task=case.task,
        expected=case.expected,
    )


def benchmark(registry, cases: List[EvalCase]) -> List[ExecutionMetrics]:
    """Run every case through the SAME code path, whatever provider backs it."""
    return [run_case_on(registry, case) for case in cases]


# ---------------------------------------------------------------------------
# Evaluation Metrics -- descriptive rollup, no scoring
# ---------------------------------------------------------------------------


def summarize(records: List[ExecutionMetrics]) -> Dict[str, Dict[str, Any]]:
    """Per-resource descriptive statistics.

    Reports what was measured (count, success rate, mean latency, mean token
    counts) and which error categories were seen. It deliberately computes no
    quality score and no cost -- those are not measurable from the records and
    would be fabricated otherwise.
    """
    buckets: Dict[str, Dict[str, Any]] = {}
    for record in records:
        bucket = buckets.setdefault(
            record.resource_id,
            {
                "provider": record.provider,
                "count": 0,
                "successes": 0,
                "latencies_ms": [],
                "prompt_tokens": [],
                "completion_tokens": [],
                "error_categories": set(),
            },
        )
        bucket["count"] += 1
        if record.success:
            bucket["successes"] += 1
        if record.latency_ms is not None:
            bucket["latencies_ms"].append(record.latency_ms)
        if record.prompt_tokens is not None:
            bucket["prompt_tokens"].append(record.prompt_tokens)
        if record.completion_tokens is not None:
            bucket["completion_tokens"].append(record.completion_tokens)
        if record.error_category:
            bucket["error_categories"].add(record.error_category)

    out: Dict[str, Dict[str, Any]] = {}
    for resource_id, bucket in buckets.items():
        out[resource_id] = {
            "provider": bucket["provider"],
            "count": bucket["count"],
            "success_rate": (
                bucket["successes"] / bucket["count"] if bucket["count"] else 0.0
            ),
            "mean_latency_ms": (
                round(sum(bucket["latencies_ms"]) / len(bucket["latencies_ms"]))
                if bucket["latencies_ms"]
                else None
            ),
            "mean_prompt_tokens": (
                round(sum(bucket["prompt_tokens"]) / len(bucket["prompt_tokens"]))
                if bucket["prompt_tokens"]
                else None
            ),
            "mean_completion_tokens": (
                round(sum(bucket["completion_tokens"]) / len(bucket["completion_tokens"]))
                if bucket["completion_tokens"]
                else None
            ),
            "error_categories": sorted(bucket["error_categories"]),
        }
    return out

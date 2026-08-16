"""Offline tests for the provider-agnostic evaluation layer (`evaluation.py`).

No network, no paid inference: every provider is either a stub run_fn (the
existing plain-`str` contract) or an HF adapter backed by a fake client. The
suite proves metric normalization, failure handling, and that ONE benchmark code
path runs existing-style and HF resources identically.

Run with `python test_evaluation.py`.
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bootstrap import install_aos_v0_shim

install_aos_v0_shim()

import providers.hf as hf_mod
from evaluation import (
    EvalCase,
    ExecutionMetrics,
    benchmark,
    classify_error,
    normalize_result,
    run_case,
    run_case_on,
    summarize,
)
from providers.errors import (
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)

_failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label} {detail}")
        _failures.append(label)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Message(content)


class _Output:
    def __init__(self, content, model="Qwen/Qwen3-30B-A3B-Instruct", usage=None):
        self.choices = [_Choice(content)]
        self.model = model
        self.usage = usage


class FakeClient:
    def __init__(self, content="ok", error=None, usage=None):
        self.content = content
        self.error = error
        self.usage = usage
        self.calls = []

    def chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return _Output(self.content, usage=self.usage)


@dataclass
class _UsageObject:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


# ---------------------------------------------------------------------------
# Metric normalization
# ---------------------------------------------------------------------------


def test_normalize_plain_string() -> None:
    print("\n[normalize] plain string result (existing capabilities/*.run contract)")
    m = normalize_result(
        raw="existing summary",
        resource_id="summarization",
        provider="groq",
        model="llama-3.3-70b-versatile",
        task="text.summarization",
        latency_ms=1234,
    )
    check("success", m.success)
    check("text captured", m.text == "existing summary", f"got {m.text!r}")
    check("provider attributed", m.provider == "groq", f"got {m.provider!r}")
    check("model attributed", m.model == "llama-3.3-70b-versatile", f"got {m.model!r}")
    check("task attributed", m.task == "text.summarization", f"got {m.task!r}")
    check("latency captured", m.latency_ms == 1234, f"got {m.latency_ms!r}")
    check("tokens not invented", m.total_tokens is None, f"got {m.total_tokens!r}")
    check("no error category", m.error_category is None)
    check("no evaluation by default", m.evaluation is None)
    check("cost never fabricated", m.cost_usd is None, f"got {m.cost_usd!r}")
    check("timestamp present", isinstance(m.timestamp, str) and m.timestamp)


def test_normalize_hf_result() -> None:
    print("\n[normalize] HFResult (structured result with token usage)")
    result = hf_mod.HFResult(
        text="HF answer",
        model="Qwen/Qwen3-8B",
        provider="huggingface",
        usage={"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12},
    )
    m = normalize_result(raw=result, resource_id="hf_qwen3_8b", latency_ms=88)
    check("success", m.success)
    check("text captured", m.text == "HF answer", f"got {m.text!r}")
    check(
        "prompt_tokens captured",
        m.prompt_tokens == 7,
        f"got {m.prompt_tokens!r}",
    )
    check(
        "completion_tokens captured",
        m.completion_tokens == 5,
        f"got {m.completion_tokens!r}",
    )
    check(
        "total_tokens captured",
        m.total_tokens == 12,
        f"got {m.total_tokens!r}",
    )
    check(
        "model from result when not passed",
        m.model == "Qwen/Qwen3-8B",
        f"got {m.model!r}",
    )
    check(
        "provider from result when not passed",
        m.provider == "huggingface",
        f"got {m.provider!r}",
    )
    check("cost never fabricated", m.cost_usd is None, f"got {m.cost_usd!r}")


def test_normalize_usage_object() -> None:
    print("\n[normalize] token usage as an object (dataclass)")
    result = hf_mod.HFResult(
        text="x", model="m", provider="p", usage=_UsageObject(1, 2, 3)
    )
    m = normalize_result(raw=result, resource_id="r")
    check(
        "object usage normalized",
        (m.prompt_tokens, m.completion_tokens, m.total_tokens) == (1, 2, 3),
        f"got {(m.prompt_tokens, m.completion_tokens, m.total_tokens)}",
    )


def test_explicit_attribution_wins() -> None:
    print("\n[normalize] explicit provider/model override the result object")
    result = hf_mod.HFResult(text="x", model="from-result", provider="from-result")
    m = normalize_result(
        raw=result, resource_id="r", provider="huggingface", model="Qwen/Qwen3-8B"
    )
    check(
        "provider from experiment, not result",
        m.provider == "huggingface",
        f"got {m.provider!r}",
    )
    check(
        "model from experiment, not result",
        m.model == "Qwen/Qwen3-8B",
        f"got {m.model!r}",
    )


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


def test_failure_handling() -> None:
    print("\n[failure] typed provider errors map to categories")
    cases = [
        (ProviderAuthenticationError("no token"), "auth"),
        (ProviderRateLimitError("429"), "rate_limit"),
        (ProviderTimeoutError("slow"), "timeout"),
        (ProviderUnavailableError("503"), "unavailable"),
        (RuntimeError("boom"), "RuntimeError"),
    ]
    for exc, category in cases:
        m = run_case(
            invoke=lambda text, instruction=None, exc=exc: (_ for _ in ()).throw(exc),
            resource_id="x",
            input_text="hi",
            provider="huggingface",
        )
        check(f"{type(exc).__name__} -> '{category}'", m.error_category == category, f"got {m.error_category!r}")
        check(f"{type(exc).__name__}: success=False", not m.success)
        check(f"{type(exc).__name__}: latency measured", m.latency_ms is not None)
        check(f"{type(exc).__name__}: no text on failure", m.text is None)


def test_classify_error_direct() -> None:
    print("\n[failure] classify_error maps base ProviderError and unknown types")
    check(
        "base ProviderError -> type name",
        classify_error(ProviderAuthenticationError("x")) == "auth",
    )
    check(
        "ValueError -> 'ValueError'",
        classify_error(ValueError("x")) == "ValueError",
    )


# ---------------------------------------------------------------------------
# run_case behavior
# ---------------------------------------------------------------------------


def test_run_case_measures_latency() -> None:
    print("\n[run_case] latency is measured, evaluation result optional")
    m = run_case(
        invoke=lambda text, instruction=None: (time.sleep(0.01), "slow ok")[1],
        resource_id="x",
        input_text="hi",
    )
    check("success", m.success)
    check("latency reflects the call", m.latency_ms is not None and m.latency_ms >= 8, f"got {m.latency_ms!r}")
    check("text captured", m.text == "slow ok", f"got {m.text!r}")


def test_run_case_expected_output() -> None:
    print("\n[run_case] caller-supplied expected output drives evaluation field")
    m = run_case(
        invoke=lambda text, instruction=None: "PONG output here",
        resource_id="x",
        input_text="hi",
        expected="PONG",
    )
    check(
        "matching expected -> passed",
        m.evaluation == {"expected": "PONG", "passed": True},
        f"got {m.evaluation!r}",
    )

    m = run_case(
        invoke=lambda text, instruction=None: "PONG output here",
        resource_id="x",
        input_text="hi",
        expected="MISSING",
    )
    check(
        "non-matching expected -> failed",
        m.evaluation == {"expected": "MISSING", "passed": False},
        f"got {m.evaluation!r}",
    )

    m = run_case(
        invoke=lambda text, instruction=None: "PONG output here",
        resource_id="x",
        input_text="hi",
    )
    check("no expected -> evaluation stays None", m.evaluation is None)


def test_failure_has_no_cost() -> None:
    print("\n[run_case] failures and successes never fabricate cost")
    ok = run_case(
        invoke=lambda text, instruction=None: hf_mod.HFResult(
            text="x", model="m", provider="huggingface", usage={"total_tokens": 9}
        ),
        resource_id="r",
        input_text="hi",
    )
    fail = run_case(
        invoke=lambda text, instruction=None: (_ for _ in ()).throw(ProviderUnavailableError("down")),
        resource_id="r",
        input_text="hi",
    )
    check("success cost None", ok.cost_usd is None, f"got {ok.cost_usd!r}")
    check("failure cost None", fail.cost_usd is None, f"got {fail.cost_usd!r}")
    check("failure still measures latency", fail.latency_ms is not None)


# ---------------------------------------------------------------------------
# One benchmark path for every provider
# ---------------------------------------------------------------------------


def _register_existing_stub(registry, resource_id, output, error=None):
    from capability_registry import CapabilityManifest

    def run_fn(text, instruction=None):
        if error is not None:
            raise error
        return output

    registry.register(
        CapabilityManifest(
            resource_id=resource_id,
            resource_class="llm",
            capabilities=["text.summarization"],
            metadata={"provider": "groq", "model": "llama-3.3-70b-versatile"},
        ),
        run_fn,
    )


def test_same_benchmark_code_for_both_providers() -> None:
    print("\n[benchmark] ONE code path executes existing and HF resources")
    from capability_registry import CapabilityRegistry

    fake = FakeClient(
        content="HF summary",
        usage={"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
    )
    registry = CapabilityRegistry()
    _register_existing_stub(registry, "summarization", "existing summary")
    hf_mod.register_hf_resources(registry, client=fake)

    hf_invoke = lambda text, instruction=None: hf_mod.run_result(  # noqa: E731
        text, instruction=instruction, client=fake, temperature=0.0
    )

    cases = [
        EvalCase("summarization", "some source", instruction="Summarize"),
        EvalCase(
            "hf_qwen3_30b_a3b_instruct",
            "some source",
            instruction="Summarize",
            invoke=hf_invoke,
        ),
    ]
    records = benchmark(registry, cases)

    check("two records produced", len(records) == 2, f"got {len(records)}")
    existing, hf = records
    check("existing record is normalized metrics", isinstance(existing, ExecutionMetrics))
    check("hf record is normalized metrics", isinstance(hf, ExecutionMetrics))
    check("existing succeeded", existing.success)
    check("hf succeeded", hf.success)
    check(
        "existing provider attributed from manifest",
        existing.provider == "groq",
        f"got {existing.provider!r}",
    )
    check(
        "hf provider attributed from manifest",
        hf.provider == "huggingface",
        f"got {hf.provider!r}",
    )
    check(
        "existing model attributed",
        existing.model == "llama-3.3-70b-versatile",
        f"got {existing.model!r}",
    )
    check(
        "hf model attributed",
        hf.model == "Qwen/Qwen3-30B-A3B-Instruct",
        f"got {hf.model!r}",
    )
    check(
        "existing tokens honestly None (str contract)",
        existing.total_tokens is None,
        f"got {existing.total_tokens!r}",
    )
    check(
        "hf tokens captured (provider exposes them)",
        hf.total_tokens == 14,
        f"got {hf.total_tokens!r}",
    )
    check("existing latency measured", existing.latency_ms is not None)
    check("hf latency measured", hf.latency_ms is not None)
    check("existing timestamp present", bool(existing.timestamp))
    check("hf timestamp present", bool(hf.timestamp))
    check("no cost fabricated", existing.cost_usd is None and hf.cost_usd is None)

    summary = summarize(records)
    check(
        "rollup success rate",
        summary["summarization"]["success_rate"] == 1.0
        and summary["hf_qwen3_30b_a3b_instruct"]["success_rate"] == 1.0,
        f"got {summary}",
    )
    check(
        "rollup mean tokens only where measured",
        summary["hf_qwen3_30b_a3b_instruct"]["mean_prompt_tokens"] == 10
        and summary["summarization"]["mean_prompt_tokens"] is None,
        f"got {summary}",
    )


def test_benchmark_default_invoke_uses_registry_run_fn() -> None:
    print("\n[benchmark] default invoke = registry.run_fn (str contract)")
    from capability_registry import CapabilityRegistry

    fake = FakeClient(content="ok")
    registry = CapabilityRegistry()
    _register_existing_stub(registry, "summarization", "existing summary")
    hf_mod.register_hf_resources(registry, client=fake)

    records = benchmark(
        registry,
        [
            EvalCase("summarization", "src"),
            EvalCase("hf_qwen3_30b_a3b_instruct", "src", instruction="Summarize"),
        ],
    )
    existing, hf = records
    check("existing via run_fn", existing.text == "existing summary", f"got {existing.text!r}")
    check(
        "hf via registry run_fn returns str -> no tokens",
        hf.success and hf.total_tokens is None,
        f"got {hf.total_tokens!r}",
    )
    check(
        "hf provider still attributed from manifest",
        hf.provider == "huggingface",
        f"got {hf.provider!r}",
    )


def test_benchmark_failure_case() -> None:
    print("\n[benchmark] failures flow through the same path")
    from capability_registry import CapabilityRegistry

    registry = CapabilityRegistry()
    _register_existing_stub(
        registry,
        "flaky",
        None,
        error=ProviderTimeoutError("inference slow"),
    )
    records = benchmark(registry, [EvalCase("flaky", "src", instruction="Go")])
    record = records[0]
    check("failure recorded", not record.success)
    check("failure categorized", record.error_category == "timeout", f"got {record.error_category!r}")
    check("failure text is None", record.text is None)
    check("failure latency measured", record.latency_ms is not None)
    check(
        "failure attribution intact",
        record.provider == "groq" and record.model == "llama-3.3-70b-versatile",
        f"got provider={record.provider!r} model={record.model!r}",
    )


def test_default_manifests_carry_provider_metadata() -> None:
    print("\n[metadata] default resources expose provider/model for attribution")
    from resource_registration import build_default_registry

    default = build_default_registry()
    check(
        "vision -> ollama",
        default.describe("vision").metadata.get("provider") == "ollama",
        f"got {default.describe('vision').metadata}",
    )
    check(
        "summarization -> groq model",
        default.describe("summarization").metadata.get("model") == "llama-3.3-70b-versatile",
        f"got {default.describe('summarization').metadata}",
    )
    check(
        "synthesis provider",
        default.describe("synthesis").metadata.get("provider") == "groq",
        f"got {default.describe('synthesis').metadata}",
    )


def test_provider_agnostic_core() -> None:
    print("\n[core] evaluation.py imports no provider implementation")
    src = (Path(__file__).resolve().parent / "evaluation.py").read_text(encoding="utf-8")
    for needle in ("providers.hf", "huggingface_hub", "InferenceClient", "Groq(", "OpenAI("):
        check(f"core avoids {needle!r}", needle not in src)
    check("core imports no capabilities.run modules", "from capabilities" not in src)


if __name__ == "__main__":
    test_normalize_plain_string()
    test_normalize_hf_result()
    test_normalize_usage_object()
    test_explicit_attribution_wins()
    test_failure_handling()
    test_classify_error_direct()
    test_run_case_measures_latency()
    test_run_case_expected_output()
    test_failure_has_no_cost()
    test_same_benchmark_code_for_both_providers()
    test_benchmark_default_invoke_uses_registry_run_fn()
    test_benchmark_failure_case()
    test_default_manifests_carry_provider_metadata()
    test_provider_agnostic_core()

    print("\n" + "=" * 60)
    if _failures:
        print(f"{len(_failures)} FAILURE(S):")
        for name in _failures:
            print(f"  - {name}")
        sys.exit(1)
    print("all checks passed")

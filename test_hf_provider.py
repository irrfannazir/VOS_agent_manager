"""Offline tests for the Hugging Face Inference Providers adapter.

No network, no paid inference: the external API is replaced by a fake client,
so the suite exercises the adapter's contract (auth, routing, normalization,
error mapping, redaction) and the registry integration (registration,
discoverability, capability-based selection, adapter invocation,
provider-agnostic agents, coexistence with the default pool) deterministically.
Run with `python test_hf_provider.py`.
"""

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bootstrap import install_aos_v0_shim

install_aos_v0_shim()

import requests
from huggingface_hub.errors import HfHubHTTPError, InferenceTimeoutError

import providers.hf as hf_mod
from providers.errors import (
    ProviderAuthenticationError,
    ProviderBadRequestError,
    ProviderError,
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


@dataclass
class _Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Message(content)


class _ChatOutput:
    def __init__(self, content, model=None, usage=None):
        self.choices = [_Choice(content)]
        self.model = model
        self.usage = usage
        self.id = "chatcmpl-test"
        self.created = 0
        self.system_fingerprint = "test-fp"


class FakeClient:
    """Stands in for `InferenceClient`. Records calls; can raise or return fixed output."""

    def __init__(self, content="ok", error=None, model=None, usage=None, output=None):
        self.content = content
        self.error = error
        self.model = model
        self.usage = usage
        self.output = output
        self.calls: list[dict] = []

    def chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        if self.output is not None:
            return self.output
        return _ChatOutput(self.content, model=self.model, usage=self.usage)


class _FakeResponse:
    """Minimal requests.Response stand-in HfHubHTTPError can wrap."""

    def __init__(self, status_code, text=""):
        self.status_code = status_code
        self.text = text
        self.headers = {"x-request-id": "test-req"}
        self.request = None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_config_loading() -> None:
    print("\n[config] env -> config mapping")
    import config as config_module

    saved = {k: os.environ.get(k) for k in ("HF_TOKEN", "HF_PROVIDER", "HF_MODEL")}
    try:
        os.environ["HF_TOKEN"] = "hf_test_token_abc"
        os.environ["HF_PROVIDER"] = "together"
        os.environ["HF_MODEL"] = "test/org-model"
        importlib.reload(config_module)
        check(
            "HF_TOKEN read from env",
            config_module.HF_TOKEN == "hf_test_token_abc",
            f"got {config_module.HF_TOKEN!r}",
        )
        check(
            "HF_PROVIDER read from env",
            config_module.HF_PROVIDER == "together",
            f"got {config_module.HF_PROVIDER!r}",
        )
        check(
            "HF_MODEL read from env",
            config_module.HF_MODEL == "test/org-model",
            f"got {config_module.HF_MODEL!r}",
        )
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        importlib.reload(config_module)


def test_config_hug_fallback() -> None:
    print("\n[config] legacy HUG key fallback")
    import config as config_module

    saved = {k: os.environ.get(k) for k in ("HF_TOKEN", "HUG")}
    try:
        os.environ.pop("HF_TOKEN", None)
        os.environ["HUG"] = "hf_legacy_fallback"
        importlib.reload(config_module)
        check(
            "HF_TOKEN falls back to HUG",
            config_module.HF_TOKEN == "hf_legacy_fallback",
            f"got {config_module.HF_TOKEN!r}",
        )
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        importlib.reload(config_module)


def test_missing_token() -> None:
    print("\n[provider] missing HF_TOKEN")
    provider = hf_mod.HFProvider(token="", client=FakeClient())
    try:
        provider.execute(messages=[{"role": "user", "content": "hi"}])
        check("missing token raises", False, "no error raised")
    except ProviderAuthenticationError as exc:
        check("missing token raises ProviderAuthenticationError", True)
        check(
            "error names HF_TOKEN",
            "HF_TOKEN" in str(exc),
            f"got {str(exc)!r}",
        )

    provider = hf_mod.HFProvider(token="hf_test", model="", client=FakeClient())
    try:
        provider.execute(messages=[{"role": "user", "content": "hi"}])
        check("no model raises", False, "no error raised")
    except ProviderError as exc:
        check("empty model raises ProviderError", True)


def test_successful_invocation() -> None:
    print("\n[provider] successful invocation")
    model = "Qwen/Qwen3-30B-A3B-Instruct"
    fake = FakeClient(
        content="The solar answer",
        model=model,
        usage=_Usage(prompt_tokens=11, completion_tokens=22, total_tokens=33),
    )
    provider = hf_mod.HFProvider(token="hf_test", model=model, provider="auto", client=fake)

    result = provider.execute(
        messages=[{"role": "user", "content": "summarize"}],
        temperature=0.5,
        max_tokens=64,
    )

    check("returns normalized text", result.text == "The solar answer", f"got {result.text!r}")
    check("returns the model id", result.model == model, f"got {result.model!r}")
    check("returns the provider", result.provider == "auto", f"got {result.provider!r}")
    check(
        "returns token usage",
        result.usage == {"prompt_tokens": 11, "completion_tokens": 22, "total_tokens": 33},
        f"got {result.usage!r}",
    )

    call = fake.calls[0]
    check("model forwarded", call["model"] == model, f"got {call.get('model')!r}")
    check(
        "messages forwarded",
        call["messages"] == [{"role": "user", "content": "summarize"}],
        f"got {call.get('messages')!r}",
    )
    check("temperature forwarded", call["temperature"] == 0.5, f"got {call.get('temperature')!r}")
    check("max_tokens forwarded", call["max_tokens"] == 64, f"got {call.get('max_tokens')!r}")


def test_provider_model_config() -> None:
    print("\n[provider] model / provider selection")
    fake = FakeClient(content="ok")
    provider = hf_mod.HFProvider(
        token="hf_test", model="custom/org-model", provider="nebius", client=fake
    )
    provider.execute(messages=[{"role": "user", "content": "hi"}])
    check(
        "custom model id forwarded",
        fake.calls[0]["model"] == "custom/org-model",
        f"got {fake.calls[0].get('model')!r}",
    )
    check("provider recorded", provider.provider == "nebius", f"got {provider.provider!r}")

    fake2 = FakeClient(content="ok")
    auto = hf_mod.HFProvider(token="hf_test", client=fake2)
    auto.execute(messages=[{"role": "user", "content": "hi"}])
    check(
        "default model resolves to DEFAULT_HF_MODEL",
        fake2.calls[0]["model"] == hf_mod.DEFAULT_HF_MODEL,
        f"got {fake2.calls[0].get('model')!r}",
    )


def test_client_construction() -> None:
    print("\n[provider] InferenceClient construction (token/provider/timeout)")
    orig = hf_mod.InferenceClient
    captured: dict = {}

    class FakeIC:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def chat_completion(self, **kwargs):
            return _ChatOutput("x", model="A/B", usage=_Usage(1, 2, 3))

    try:
        hf_mod.InferenceClient = FakeIC

        provider = hf_mod.HFProvider(
            token="hf_secret_xyz", model="A/B", provider="auto", timeout=42.0
        )
        provider.execute(messages=[{"role": "user", "content": "hi"}])
        check(
            "client built with the token",
            captured.get("token") == "hf_secret_xyz",
            f"got {captured.get('token')!r}",
        )
        check(
            "client built with provider routing",
            captured.get("provider") == "auto",
            f"got {captured.get('provider')!r}",
        )
        check(
            "client built with timeout",
            captured.get("timeout") == 42.0,
            f"got {captured.get('timeout')!r}",
        )
        check(
            "default router URL not overridden",
            "base_url" not in captured,
            f"got base_url={captured.get('base_url')!r}",
        )

        captured.clear()
        custom = hf_mod.HFProvider(
            token="hf_test", model="A/B", base_url="https://custom.example/v1"
        )
        custom.execute(messages=[{"role": "user", "content": "hi"}])
        check(
            "custom base_url forwarded",
            captured.get("base_url") == "https://custom.example/v1",
            f"got {captured.get('base_url')!r}",
        )
    finally:
        hf_mod.InferenceClient = orig


def test_auth_failure() -> None:
    print("\n[provider] authentication failure (401)")
    err = HfHubHTTPError("401 Unauthorized", _FakeResponse(401, "invalid token"))
    fake = FakeClient(error=err)
    provider = hf_mod.HFProvider(token="hf_bad", client=fake)
    try:
        provider.execute(messages=[{"role": "user", "content": "hi"}])
        check("401 raises", False, "no error raised")
    except ProviderAuthenticationError as exc:
        check("401 -> ProviderAuthenticationError", True)
        check("token not leaked", "hf_bad" not in str(exc), f"got {str(exc)!r}")


def test_rate_limit_and_bad_request() -> None:
    print("\n[provider] rate limit (429) and bad request (404)")
    fake = FakeClient(error=HfHubHTTPError("429 Too Many Requests", _FakeResponse(429)))
    provider = hf_mod.HFProvider(token="hf_test", client=fake)
    try:
        provider.execute(messages=[{"role": "user", "content": "hi"}])
        check("429 raises", False, "no error raised")
    except ProviderRateLimitError:
        check("429 -> ProviderRateLimitError", True)

    fake = FakeClient(error=HfHubHTTPError("404 Not Found", _FakeResponse(404)))
    provider = hf_mod.HFProvider(token="hf_test", client=fake)
    try:
        provider.execute(messages=[{"role": "user", "content": "hi"}])
        check("404 raises", False, "no error raised")
    except ProviderBadRequestError:
        check("404 -> ProviderBadRequestError", True)


def test_provider_failure() -> None:
    print("\n[provider] provider failure (503)")
    fake = FakeClient(error=HfHubHTTPError("503 Service Unavailable", _FakeResponse(503)))
    provider = hf_mod.HFProvider(token="hf_test", client=fake)
    try:
        provider.execute(messages=[{"role": "user", "content": "hi"}])
        check("503 raises", False, "no error raised")
    except ProviderUnavailableError:
        check("503 -> ProviderUnavailableError", True)


def test_timeout_and_network_failure() -> None:
    print("\n[provider] timeout / network failures")

    for label, err, expected in [
        ("InferenceTimeoutError", InferenceTimeoutError("Inference call timed out"), ProviderTimeoutError),
        ("requests Timeout", requests.exceptions.Timeout("timed out"), ProviderTimeoutError),
        ("builtin TimeoutError", TimeoutError("timed out"), ProviderTimeoutError),
        ("requests ConnectionError", requests.exceptions.ConnectionError("connection refused"), ProviderUnavailableError),
    ]:
        fake = FakeClient(error=err)
        provider = hf_mod.HFProvider(token="hf_test", client=fake)
        try:
            provider.execute(messages=[{"role": "user", "content": "hi"}])
            check(f"{label} raises", False, "no error raised")
        except expected:
            check(f"{label} -> {expected.__name__}", True)
        except ProviderError as exc:  # noqa: PERF203
            check(f"{label} -> {expected.__name__}", False, f"got {type(exc).__name__}")


def test_response_normalization() -> None:
    print("\n[provider] response normalization")

    provider = hf_mod.HFProvider(token="hf_test", client=FakeClient(content=None))
    result = provider.execute(messages=[{"role": "user", "content": "hi"}])
    check("None content -> empty text", result.text == "", f"got {result.text!r}")
    check("no usage -> None", result.usage is None, f"got {result.usage!r}")

    empty_choices = hf_mod.HFProvider(
        token="hf_test",
        client=FakeClient(output=object()),  # no .choices at all
    )
    result = empty_choices.execute(messages=[{"role": "user", "content": "hi"}])
    check("missing choices -> empty text", result.text == "", f"got {result.text!r}")

    dict_usage = hf_mod.HFProvider(
        token="hf_test",
        client=FakeClient(content="x", usage={"total_tokens": 7}),
    )
    result = dict_usage.execute(messages=[{"role": "user", "content": "hi"}])
    check(
        "dict usage normalized",
        result.usage == {"total_tokens": 7},
        f"got {result.usage!r}",
    )


def test_secret_redaction() -> None:
    print("\n[provider] secret redaction")
    token = "hf_secret_123456"

    check(
        "redact helper scrubs the token",
        hf_mod._redact(f"token={token} bad", token) == "token=*** bad",
    )

    err = RuntimeError(f"Authorization: Bearer {token} rejected")
    provider = hf_mod.HFProvider(token=token, client=FakeClient(error=err))
    try:
        provider.execute(messages=[{"role": "user", "content": "hi"}])
        check("raises", False, "no error raised")
    except ProviderError as exc:
        check("typed error raised", isinstance(exc, ProviderError))
        check("token redacted from generic error", token not in str(exc), f"got {str(exc)!r}")
        check("redaction placeholder present", "***" in str(exc), f"got {str(exc)!r}")

    http_err = HfHubHTTPError(f"401 {token}", _FakeResponse(401, f"token {token} invalid"))
    provider2 = hf_mod.HFProvider(token=token, client=FakeClient(error=http_err))
    try:
        provider2.execute(messages=[{"role": "user", "content": "hi"}])
        check("http error raises", False, "no error raised")
    except ProviderAuthenticationError as exc:
        check("token redacted from http error", token not in str(exc), f"got {str(exc)!r}")


def test_run_wrapper() -> None:
    print("\n[provider] capabilities/*.run compatibility wrapper")
    fake = FakeClient(content="HF summary")
    out = hf_mod.run("lots of source text", instruction="Summarize this", client=fake, temperature=0.1)
    check("returns plain text", out == "HF summary", f"got {out!r}")

    call = fake.calls[0]
    messages = call["messages"]
    check("generic run has no system prompt by default", len(messages) == 1, f"got {messages!r}")
    check(
        "instruction augments the user message",
        messages[0]["content"].startswith("Summarize this"),
        f"got {messages[0]['content']!r}",
    )
    check("generation params forwarded", call["temperature"] == 0.1, f"got {call.get('temperature')!r}")

    fake2 = FakeClient(content="with system")
    hf_mod.run(
        "text",
        instruction="Summarize",
        system="You are a summarizer.",
        client=fake2,
    )
    messages2 = fake2.calls[0]["messages"]
    check(
        "explicit system prompt prepended",
        len(messages2) == 2
        and messages2[0] == {"role": "system", "content": "You are a summarizer."},
        f"got {messages2!r}",
    )


CATALOG_IDS = [
    "hf_qwen3_30b_a3b_instruct",
    "hf_qwen3_8b",
    "hf_qwen2_5_vl_7b_instruct",
    "hf_qwen3_coder_30b_a3b",
    "hf_deepseek_r1_distill_qwen_32b",
    "hf_llama3_3_70b_instruct",
    "hf_llama3_2_3b_instruct",
    "hf_gemma2_27b_it",
    "hf_bge_large_en_v1_5",
    "hf_bge_reranker_v2_m3",
    "hf_whisper_large_v3_turbo",
    # Multimodal / specialized models
    "hf_whisper_large_v3",
    "hf_qwen2_audio_7b_instruct",
    "hf_ast_audioset_finetuned",
    "hf_siglip2_base_224",
    "hf_yolo11",
    "hf_dinov3_vitb16",
]

CHAT_MODEL_IDS = [
    "hf_qwen3_30b_a3b_instruct",
    "hf_qwen3_8b",
    "hf_qwen2_5_vl_7b_instruct",
    "hf_qwen3_coder_30b_a3b",
    "hf_deepseek_r1_distill_qwen_32b",
    "hf_llama3_3_70b_instruct",
    "hf_llama3_2_3b_instruct",
    "hf_gemma2_27b_it",
]


def _loose() -> "DNAConstraints":
    from models import DNAConstraints

    return DNAConstraints(cost_ceiling_usd=0.05, latency_slo_ms=30_000)


def test_registration() -> None:
    print("\n[1] HF resources can be registered")
    from capability_registry import CapabilityRegistry, LatencyModel

    registry = CapabilityRegistry()
    manifests = hf_mod.register_hf_resources(
        registry, provider="auto", client=FakeClient(content="ok")
    )

    check(
        "one manifest per catalog model",
        {m.resource_id for m in manifests} == set(CATALOG_IDS),
        f"got {[m.resource_id for m in manifests]}",
    )
    check(
        "manifests registered on the registry",
        {m.resource_id for m in registry.manifests()} == set(CATALOG_IDS),
    )

    spec_by_id = {s.resource_id: s for s in hf_mod.HF_MODEL_CATALOG}
    for m in manifests:
        spec = spec_by_id[m.resource_id]
        meta = m.metadata
        check(f"{m.resource_id}: class matches spec", m.resource_class == spec.resource_class)
        check(
            f"{m.resource_id}: declared capabilities",
            set(m.capabilities) == set(spec.capabilities),
            f"got {m.capabilities}",
        )
        check(f"{m.resource_id}: metadata.model", meta.get("model") == spec.model, f"got {meta!r}")
        check(
            f"{m.resource_id}: metadata.provider",
            meta.get("provider") == "huggingface",
            f"got {meta!r}",
        )
        check(f"{m.resource_id}: metadata.task", meta.get("task") == spec.task, f"got {meta!r}")
        check(
            f"{m.resource_id}: metadata.hf_provider",
            meta.get("hf_provider") == "auto",
            f"got {meta!r}",
        )
        check(
            f"{m.resource_id}: metadata.interface",
            meta.get("interface") == spec.interface,
            f"got {meta!r}",
        )
        _wired = {"chat_completion", "automatic_speech_recognition", "audio_chat_completion", "audio_classification"}
        check(
            f"{m.resource_id}: transport wired iff chat/asr/audio",
            meta.get("transport") == ("wired" if spec.interface in _wired else "declared"),
            f"got {meta!r}",
        )
        check(
            f"{m.resource_id}: model card source recorded",
            meta.get("source", "").startswith("https://huggingface.co/"),
            f"got {meta!r}",
        )
        check(f"{m.resource_id}: io declared", m.input_schema.type == spec.input_type)
        check(f"{m.resource_id}: no quality claims", m.quality_priors == {})
        check(
            f"{m.resource_id}: no fabricated cost",
            m.cost_model.estimate_usd == 0.0,
            f"got {m.cost_model.estimate_usd!r}",
        )
        check(
            f"{m.resource_id}: no fabricated latency",
            m.latency_model == LatencyModel(),
            f"got {m.latency_model!r}",
        )

    import json

    for m in manifests:
        prov = json.loads(m.metadata["capability_provenance"])
        check(
            f"{m.resource_id}: provenance covers exactly its capabilities",
            set(prov) == set(m.capabilities)
            and set(prov.values()) <= {"documented", "inferred"},
            f"got {prov}",
        )

    check(
        "vlm model registered as vlm class",
        registry.describe("hf_qwen2_5_vl_7b_instruct").resource_class == "vlm",
    )
    check(
        "text models registered as llm class",
        registry.describe("hf_qwen3_30b_a3b_instruct").resource_class == "llm",
    )
    check(
        "whisper large-v3 registered as asr class",
        registry.describe("hf_whisper_large_v3").resource_class == "asr",
    )
    check(
        "qwen2-audio registered as audio class",
        registry.describe("hf_qwen2_audio_7b_instruct").resource_class == "audio",
    )
    check(
        "AST registered as audio class",
        registry.describe("hf_ast_audioset_finetuned").resource_class == "audio",
    )
    check(
        "siglip2 registered as image class",
        registry.describe("hf_siglip2_base_224").resource_class == "image",
    )
    check(
        "yolo11 registered as image class",
        registry.describe("hf_yolo11").resource_class == "image",
    )
    check(
        "dinov3 registered as image class",
        registry.describe("hf_dinov3_vitb16").resource_class == "image",
    )


def test_discoverability() -> None:
    print("\n[2] their capabilities are discoverable")
    from capability_registry import CapabilityRegistry
    from models import CapabilityDNA

    registry = CapabilityRegistry()
    hf_mod.register_hf_resources(registry, client=FakeClient(content="ok"))

    provided = set(registry.provided_flags())
    for flag in (
        "text.summarization",
        "reasoning.deep",
        "answer.synthesis",
        "vision.understanding",
        "text.classification",
        "reasoning.shallow",
        "code.generation",
        "tool.calling",
        "embedding.generation",
        "rerank.scoring",
        "speech.transcription",
        "audio_input",
        "speech_recognition",
        "audio_understanding",
        "audio_classification",
        "vision_input",
        "image_classification",
        "object_detection",
        "visual_reasoning",
        "visual_embedding",
        "instruction_following",
    ):
        check(f"'{flag}' discoverable via provided_flags", flag in provided)

    check(
        "find(['vision.understanding']) -> the vlm model",
        registry.find(["vision.understanding"]) == ["hf_qwen2_5_vl_7b_instruct"],
        f"got {registry.find(['vision.understanding'])}",
    )
    check(
        "find(['embedding.generation']) -> the embedding model",
        registry.find(["embedding.generation"]) == ["hf_bge_large_en_v1_5"],
        f"got {registry.find(['embedding.generation'])}",
    )
    check(
        "find(['rerank.scoring']) -> the reranker",
        registry.find(["rerank.scoring"]) == ["hf_bge_reranker_v2_m3"],
        f"got {registry.find(['rerank.scoring'])}",
    )
    check(
        "find(['speech.transcription']) -> both ASR models",
        set(registry.find(["speech.transcription"]))
        == {"hf_whisper_large_v3", "hf_whisper_large_v3_turbo"},
        f"got {registry.find(['speech.transcription'])}",
    )
    check(
        "find(['audio_input']) -> all four audio models",
        set(registry.find(["audio_input"]))
        == {
            "hf_whisper_large_v3",
            "hf_whisper_large_v3_turbo",
            "hf_qwen2_audio_7b_instruct",
            "hf_ast_audioset_finetuned",
        },
        f"got {registry.find(['audio_input'])}",
    )
    check(
        "find(['vision_input']) -> all five vision models",
        set(registry.find(["vision_input"]))
        == {
            "hf_qwen2_5_vl_7b_instruct",
            "hf_siglip2_base_224",
            "hf_yolo11",
            "hf_dinov3_vitb16",
            # gemma2 etc. are text-only; llm are text-only
        },
        f"got {registry.find(['vision_input'])}",
    )
    check(
        "find(['text.summarization']) -> every general text model",
        set(registry.find(["text.summarization"]))
        == {
            "hf_qwen3_30b_a3b_instruct",
            "hf_qwen3_8b",
            "hf_llama3_3_70b_instruct",
            "hf_llama3_2_3b_instruct",
            "hf_gemma2_27b_it",
        },
        f"got {registry.find(['text.summarization'])}",
    )
    check(
        "find(['reasoning.deep']) -> the reasoning-capable models",
        set(registry.find(["reasoning.deep"]))
        == {
            "hf_qwen3_30b_a3b_instruct",
            "hf_qwen3_coder_30b_a3b",
            "hf_deepseek_r1_distill_qwen_32b",
            "hf_llama3_3_70b_instruct",
            "hf_gemma2_27b_it",
        },
        f"got {registry.find(['reasoning.deep'])}",
    )
    check(
        "find(['code.generation']) -> coding-capable models",
        set(registry.find(["code.generation"]))
        == {
            "hf_qwen3_coder_30b_a3b",
            "hf_llama3_3_70b_instruct",
            "hf_gemma2_27b_it",
        },
        f"got {registry.find(['code.generation'])}",
    )
    check(
        "multi-capability find({code.generation, tool.calling, reasoning.deep})",
        set(
            registry.find(
                ["code.generation", "tool.calling", "reasoning.deep"]
            )
        )
        == {"hf_qwen3_coder_30b_a3b", "hf_llama3_3_70b_instruct"},
        f"got {registry.find(['code.generation', 'tool.calling', 'reasoning.deep'])}",
    )
    check(
        "unsatisfiable flag is reported, not guessed",
        registry.unsatisfiable_flags(CapabilityDNA(flags=["web.search"])) == ["web.search"],
    )
    check(
        "db.query has no provider in the HF pool",
        registry.find(["db.query"]) == [],
        f"got {registry.find(['db.query'])}",
    )


def test_capability_selection() -> None:
    print("\n[3] capability-based lookup selects an HF resource")
    from capability_registry import CapabilityRegistry
    from models import CapabilityDNA

    registry = CapabilityRegistry()
    hf_mod.register_hf_resources(registry, client=FakeClient(content="ok"))

    decision = registry.select(
        CapabilityDNA(flags=["vision.understanding"], constraints=_loose())
    )
    check(
        "vision requirement binds the HF vlm model",
        decision.resource_id == "hf_qwen2_5_vl_7b_instruct",
        f"got {decision.resource_id}",
    )
    check("decision is a scored audit record", len(decision.candidates) >= 1 and decision.score <= 1.0)

    decision = registry.select(
        CapabilityDNA(flags=["reasoning.deep"], constraints=_loose())
    )
    check(
        "reasoning.deep requirement binds a reasoning model "
        "(deterministic tie-break among neutral-prior candidates)",
        decision.resource_id == "hf_deepseek_r1_distill_qwen_32b",
        f"got {decision.resource_id}",
    )

    decision = registry.select(
        CapabilityDNA(flags=["text.summarization"], constraints=_loose())
    )
    check(
        "summarization requirement binds a general text model",
        decision.resource_id in {
            "hf_qwen3_30b_a3b_instruct",
            "hf_qwen3_8b",
            "hf_llama3_3_70b_instruct",
            "hf_llama3_2_3b_instruct",
            "hf_gemma2_27b_it",
        },
        f"got {decision.resource_id}",
    )

    decision = registry.select(
        CapabilityDNA(flags=["embedding.generation"], constraints=_loose())
    )
    check(
        "embedding requirement binds the embedding model",
        decision.resource_id == "hf_bge_large_en_v1_5",
        f"got {decision.resource_id}",
    )

    decision = registry.select(
        CapabilityDNA(flags=["code.generation"], constraints=_loose())
    )
    check(
        "coding requirement binds a coding-capable model",
        decision.resource_id
        in {
            "hf_qwen3_coder_30b_a3b",
            "hf_llama3_3_70b_instruct",
            "hf_gemma2_27b_it",
        },
        f"got {decision.resource_id}",
    )

    decision = registry.select(
        CapabilityDNA(
            flags=["code.generation", "tool.calling", "reasoning.deep"],
            constraints=_loose(),
        )
    )
    check(
        "multi-capability DNA narrows to the coding-agent models",
        decision.resource_id
        in {"hf_qwen3_coder_30b_a3b", "hf_llama3_3_70b_instruct"},
        f"got {decision.resource_id}",
    )
    check(
        "gemma (no tool.calling) filtered out of the multi-cap set",
        "hf_gemma2_27b_it" not in [c.resource_id for c in decision.candidates],
        f"got {[c.resource_id for c in decision.candidates]}",
    )


def test_invokes_adapter() -> None:
    print("\n[4] the selected resource invokes the HF adapter")
    from capability_registry import CapabilityRegistry
    from models import CapabilityDNA

    fake = FakeClient(content="mock HF output")
    registry = CapabilityRegistry()
    hf_mod.register_hf_resources(registry, client=fake)

    decision, run_fn = registry.bind(
        CapabilityDNA(flags=["reasoning.deep"], constraints=_loose())
    )
    expected_model = {
        s.resource_id: s.model for s in hf_mod.HF_MODEL_CATALOG
    }[decision.resource_id]
    check(
        "bind selects a reasoning-capable HF chat model",
        decision.resource_id == "hf_deepseek_r1_distill_qwen_32b",
        f"got {decision.resource_id}",
    )

    out = run_fn("some evidence", instruction="Compare and conclude")
    check("run_fn returns adapter text", out == "mock HF output", f"got {out!r}")
    check("adapter invoked exactly once", len(fake.calls) == 1, f"got {len(fake.calls)}")

    call = fake.calls[0]
    check(
        "adapter called with the bound model",
        call["model"] == expected_model,
        f"got {call.get('model')!r}",
    )
    check(
        "system + user messages built",
        len(call["messages"]) == 2 and call["messages"][0]["role"] == "system",
        f"got {call.get('messages')!r}",
    )


def test_declared_only_resources() -> None:
    print("\n[4b] declared-only resources are registered but explicitly not wired")
    from capability_registry import CapabilityRegistry

    registry = CapabilityRegistry()
    hf_mod.register_hf_resources(registry, client=FakeClient(content="ok"))

    for rid in (
        "hf_bge_large_en_v1_5",
        "hf_bge_reranker_v2_m3",
        "hf_siglip2_base_224",
        "hf_yolo11",
        "hf_dinov3_vitb16",
    ):
        manifest = registry.describe(rid)
        meta = manifest.metadata
        check(f"{rid}: transport declared", meta.get("transport") == "declared", f"got {meta!r}")
        check(
            f"{rid}: interface recorded",
            meta.get("interface") in {
                "feature_extraction",
                "rerank",
                "zero_shot_image_classification",
                "object_detection",
                "image_feature_extraction",
            },
            f"got {meta!r}",
        )
        try:
            registry.run_fn(rid)("transcript or query")
            check(f"{rid}: invocation raises", False, "no error raised")
        except ProviderError as exc:
            check(
                f"{rid}: invocation raises a typed error naming the interface",
                meta.get("interface") in str(exc),
                f"got {str(exc)!r}",
            )


def test_wired_audio_resources() -> None:
    print("\n[4c] audio resources are wired to real HF transport")
    from capability_registry import CapabilityRegistry

    registry = CapabilityRegistry()
    hf_mod.register_hf_resources(registry, client=FakeClient(content="ok"))

    # ASR models (whisper) — transport wired, run_fn validates file input.
    for rid in ("hf_whisper_large_v3_turbo", "hf_whisper_large_v3"):
        manifest = registry.describe(rid)
        meta = manifest.metadata
        check(f"{rid}: transport wired", meta.get("transport") == "wired", f"got {meta!r}")
        check(f"{rid}: interface is ASR", meta.get("interface") == "automatic_speech_recognition", f"got {meta!r}")
        try:
            registry.run_fn(rid)("not_a_real_file.wav")
            check(f"{rid}: raises on missing file", False, "no error raised")
        except ProviderError as exc:
            check(f"{rid}: error names missing file", "not a valid file" in str(exc), f"got {str(exc)!r}")

    # Audio chat model (qwen2-audio) — transport wired.
    rid = "hf_qwen2_audio_7b_instruct"
    manifest = registry.describe(rid)
    meta = manifest.metadata
    check(f"{rid}: transport wired", meta.get("transport") == "wired", f"got {meta!r}")
    check(f"{rid}: interface is audio_chat_completion", meta.get("interface") == "audio_chat_completion", f"got {meta!r}")
    try:
        registry.run_fn(rid)("not_a_real_file.wav")
        check(f"{rid}: raises on missing file", False, "no error raised")
    except ProviderError as exc:
        check(f"{rid}: error names missing file", "not a valid file" in str(exc), f"got {str(exc)!r}")

    # Audio classification model (AST) — transport wired.
    rid = "hf_ast_audioset_finetuned"
    manifest = registry.describe(rid)
    meta = manifest.metadata
    check(f"{rid}: transport wired", meta.get("transport") == "wired", f"got {meta!r}")
    check(f"{rid}: interface is audio_classification", meta.get("interface") == "audio_classification", f"got {meta!r}")
    try:
        registry.run_fn(rid)("not_a_real_file.wav")
        check(f"{rid}: raises on missing file", False, "no error raised")
    except ProviderError as exc:
        check(f"{rid}: error names missing file", "not a valid file" in str(exc), f"got {str(exc)!r}")


def test_agent_agnostic() -> None:
    print("\n[5] agents need no provider-specific logic")
    from capability_registry import CapabilityRegistry
    from models import CapabilityDNA
    from resource_registration import build_hf_enabled_registry

    # An agent / graph executor talks to the SAME registry API regardless of
    # which provider a resource uses -- there is no provider check anywhere.
    mixed = build_hf_enabled_registry()

    vision_decision = mixed.select(
        CapabilityDNA(flags=["vision.understanding"], constraints=_loose())
    )
    check(
        "mixed registry routes vision via the generic select() API",
        vision_decision.resource_id == "vision",
        f"got {vision_decision.resource_id}",
    )
    check(
        "HF vlm is a feasible candidate in the mixed registry",
        "hf_qwen2_5_vl_7b_instruct" in [c.resource_id for c in vision_decision.candidates],
        f"got {[c.resource_id for c in vision_decision.candidates]}",
    )

    hf_only = CapabilityRegistry()
    hf_mod.register_hf_resources(hf_only, client=FakeClient(content="ok"))
    decision = hf_only.select(
        CapabilityDNA(flags=["vision.understanding"], constraints=_loose())
    )
    check(
        "same call binds an HF resource when only HF is registered",
        decision.resource_id == "hf_qwen2_5_vl_7b_instruct",
        f"got {decision.resource_id}",
    )

    # Source-level proof: the agent / executor layer contains no HF symbols.
    needles = ("huggingface", "hf_qwen", "InferenceClient", "HF_TOKEN")
    targets = list((Path(__file__).resolve().parent / "agents").glob("*.py"))
    targets.append(Path(__file__).resolve().parent / "agents" / "graph_executor.py")
    bad = []
    for path in targets:
        text = path.read_text(encoding="utf-8")
        for needle in needles:
            if needle in text:
                bad.append(f"{path.name}:{needle}")
    check("agent/executor source has no HF references", not bad, f"got {bad}")


def test_non_hf_resources_continue() -> None:
    print("\n[6] existing non-HF resources keep working")
    from capability_registry import CapabilityRegistry
    from models import CapabilityDNA
    from resource_registration import build_default_registry, build_hf_enabled_registry

    default = build_default_registry()
    rids = {m.resource_id for m in default.manifests()}
    check(
        "default pool unchanged",
        rids == {"web_search", "summarization", "vision", "synthesis", "quick_summarization"},
        f"got {sorted(rids)}",
    )
    check("default pool has no HF resources", not any(r.startswith("hf_") for r in rids))

    # Exact-match fallback (nodes without DNA) still binds the default
    # summarization run_fn -- HF resources are not in that lookup path.
    check(
        "exact-match fallback unchanged",
        default.find_by_capability("summarization") is default.run_fn("summarization"),
    )

    mixed = build_hf_enabled_registry()
    summary_decision = mixed.select(
        CapabilityDNA(flags=["text.summarization"], constraints=_loose())
    )
    check(
        "default summarizer still wins on declared quality",
        summary_decision.resource_id == "summarization",
        f"got {summary_decision.resource_id}",
    )
    check(
        "HF summarizer is still a feasible candidate",
        "hf_qwen3_30b_a3b_instruct" in [c.resource_id for c in summary_decision.candidates],
        f"got {[c.resource_id for c in summary_decision.candidates]}",
    )
    check(
        "default pool present in mixed registry",
        rids <= {m.resource_id for m in mixed.manifests()},
    )
    check(
        "web_search still routes",
        mixed.select(CapabilityDNA(flags=["web.search"], constraints=_loose())).resource_id
        == "web_search",
    )


if __name__ == "__main__":
    test_config_loading()
    test_config_hug_fallback()
    test_missing_token()
    test_successful_invocation()
    test_provider_model_config()
    test_client_construction()
    test_auth_failure()
    test_rate_limit_and_bad_request()
    test_provider_failure()
    test_timeout_and_network_failure()
    test_response_normalization()
    test_secret_redaction()
    test_run_wrapper()
    test_registration()
    test_discoverability()
    test_capability_selection()
    test_invokes_adapter()
    test_declared_only_resources()
    test_agent_agnostic()
    test_non_hf_resources_continue()

    print("\n" + "=" * 60)
    if _failures:
        print(f"{len(_failures)} FAILURE(S):")
        for name in _failures:
            print(f"  - {name}")
        sys.exit(1)
    print("all checks passed")

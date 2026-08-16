"""Hugging Face Inference Providers adapter.

One provider, plugged into the Capability Registry as ordinary
(CapabilityManifest, run_fn) pairs -- the same seam every resource uses, so
nothing in the core knows Hugging Face exists.

Two entry points:

  * `HFProvider.execute(model=..., messages=..., **params)` -- the clean,
    provider-facing interface. Returns a normalized `HFResult` carrying the
    generated text plus metadata (model / provider / token usage) and raises
    typed errors from `providers.errors` instead of leaking SDK exceptions.
  * `run(text, instruction=None, ...)` -- the compatibility wrapper matching
    the `capabilities/*.run` convention the GraphExecutor / FailureManager
    expect (returns a plain `str`). `run_result(...)` is its sibling for the
    evaluation layer: identical call shape, but returns the full `HFResult` so
    provider-exposed metrics (token usage) survive the boundary.

Registration:

  * `register_hf_resources(registry)` -- registers one resource per model in
    the model catalog (`HF_MODEL_CATALOG`) through the public
    `CapabilityRegistry.register()` API. Each model is declared as a normal
    resource (CapabilityManifest) exposing its capability flags; scheduling
    routes on `capabilities` alone, so nothing in the core ever checks for the
    Hugging Face provider. Opt-in: the default pool in
    `resource_registration.build_default_registry()` is untouched, so the DOC2
    baseline control group stays frozen.

  The catalog covers seventeen real models across distinct capability classes:
  general text generation, reasoning, coding, vision-language, embeddings,
  reranking, speech transcription, audio understanding, sound classification,
  image classification, object detection, and visual representation. Every
  provenance marker in the manifest metadata (`documented` = asserted on the
  model card, `inferred` = implied by documented properties) so no entry
  fabricates a capability. Chat-completion models are wired to the adapter;
  the embedding / rerank / ASR entries are declared interfaces whose run_fn
  raises a typed error if invoked (their transports are not wired to the text
  pipeline, mirroring the VLM's image-input precedent).

Auth: the token is read from `config.HF_TOKEN` (falling back to the legacy
`HUG` key), or passed explicitly. It is never hard-coded, never logged, and is
redacted from every error message before raising.
"""

from dataclasses import asdict, dataclass
from functools import partial
from typing import Any, Dict, List, Optional

import requests

from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError, InferenceTimeoutError

from providers.errors import (
    ProviderAuthenticationError,
    ProviderBadRequestError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)

# Default model when neither HF_MODEL nor an explicit model is given. Broadly
# served across Inference Providers; cheap enough that the Pareto scorer can
# genuinely weigh it against the Groq pool rather than auto-selecting it.
DEFAULT_HF_MODEL = "Qwen/Qwen3-30B-A3B-Instruct"

# Route through the Inference Providers router unless a custom base_url is set.
DEFAULT_BASE_URL = "https://router.huggingface.co/v1"

DEFAULT_TIMEOUT_S = 60.0

# chat_completion kwargs the adapter forwards verbatim. Anything else passed in
# goes to `extra_body`, so callers can set provider-specific payload fields.
_CHAT_PARAMS = {
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "max_tokens",
    "n",
    "presence_penalty",
    "response_format",
    "seed",
    "stop",
    "stream_options",
    "temperature",
    "tool_choice",
    "tools",
    "top_logprobs",
    "top_p",
    "extra_body",
}

_AUTH_STATUSES = {401, 403}
_RATE_LIMIT_STATUSES = {429}
_BAD_REQUEST_STATUSES = {400, 404, 422}


def _redact(message: str, token: Optional[str]) -> str:
    """Replace any occurrence of the token in a message with a placeholder.

    HF responses occasionally echo the Authorization header back in error text
    (server_message / request bodies), so errors must be scrubbed before they
    become ProviderErrors or log lines.
    """
    if token:
        message = message.replace(token, "***")
    return message


@dataclass
class HFResult:
    """Normalized output of one `HFProvider.execute` call.

    `text` is what the pipeline consumes; the rest is audit metadata (mirrors
    what the Learning Manager's Experience Records want to persist).
    """

    text: str
    model: Optional[str] = None
    provider: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    raw: Any = None


class HFProvider:
    """Executes chat-completion calls against Hugging Face Inference Providers.

    `token`/`model`/`provider` default to the values in `config`; each can be
    overridden explicitly. `client` is an injection point for tests (mirrors
    the `DNAExtractor(client=...)` pattern); when omitted, a real
    `InferenceClient` is built lazily on the first `execute` call.
    """

    def __init__(
        self,
        *,
        token: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        timeout: Optional[float] = None,
        base_url: Optional[str] = None,
        client: Any = None,
    ) -> None:
        # Read lazily at construction time so tests that reload config (or
        # switch env vars) get fresh defaults.
        from config import (
            HF_MODEL as _CFG_MODEL,
            HF_PROVIDER as _CFG_PROVIDER,
            HF_TOKEN as _CFG_TOKEN,
        )

        self.token = token if token is not None else _CFG_TOKEN
        self.model = model if model is not None else (_CFG_MODEL or DEFAULT_HF_MODEL)
        self.provider = provider if provider is not None else _CFG_PROVIDER
        self.timeout = timeout if timeout is not None else DEFAULT_TIMEOUT_S
        self.base_url = base_url or DEFAULT_BASE_URL
        self._client = client

    # -- public API ---------------------------------------------------------

    def execute(
        self,
        model: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **generation_parameters: Any,
    ) -> HFResult:
        """One normalized chat-completion call.

        Raises typed ProviderError subclasses on failure; the message never
        contains the token.
        """
        if not self.token:
            raise ProviderAuthenticationError(
                "Hugging Face inference requires a token: set HF_TOKEN "
                "(or HUG) in the environment or .env"
            )

        model_id = model or self.model
        if not model_id:
            raise ProviderError("no Hugging Face model configured (set HF_MODEL)")

        if not messages:
            raise ProviderError("messages are required for chat completion")

        client = self._client or self._build_client()

        chat_kwargs: Dict[str, Any] = {}
        extra_body: Dict[str, Any] = {}
        for key, value in generation_parameters.items():
            (chat_kwargs if key in _CHAT_PARAMS else extra_body)[key] = value
        if extra_body:
            chat_kwargs["extra_body"] = extra_body

        try:
            output = client.chat_completion(
                messages=messages, model=model_id, **chat_kwargs
            )
        # InferenceTimeoutError subclasses BOTH requests.HTTPError and
        # TimeoutError, so the timeout clause must come before the HTTP clause.
        except (InferenceTimeoutError, TimeoutError, requests.exceptions.Timeout) as exc:
            raise ProviderTimeoutError(
                _redact(f"HF inference timed out: {exc}", self.token)
            ) from exc
        except (HfHubHTTPError, requests.HTTPError) as exc:
            raise self._map_http_error(exc) from exc
        except requests.exceptions.RequestException as exc:
            raise ProviderUnavailableError(
                _redact(f"HF inference network failure: {exc}", self.token)
            ) from exc
        except ProviderError:
            raise
        except Exception as exc:  # noqa: BLE001 -- every failure becomes a typed error
            raise ProviderError(_redact(f"HF inference failed: {exc}", self.token)) from exc

        text = self._extract_text(output)
        return HFResult(
            text=text,
            model=getattr(output, "model", None) or model_id,
            provider=self.provider,
            usage=_usage_to_dict(getattr(output, "usage", None)),
            raw=output,
        )

    # -- internals ----------------------------------------------------------

    def _build_client(self) -> InferenceClient:
        kwargs: Dict[str, Any] = {
            "token": self.token,
            "provider": self.provider,
            "timeout": self.timeout,
        }
        if self.base_url != DEFAULT_BASE_URL:
            # Only override the router URL when the caller actually asked for a
            # custom endpoint; otherwise let the SDK pick the router it knows.
            kwargs["base_url"] = self.base_url
        return InferenceClient(**kwargs)

    def _map_http_error(self, exc) -> ProviderError:
        """Map an SDK HTTP error to a typed ProviderError by status code."""
        status = getattr(getattr(exc, "response", None), "status_code", None)
        message = _redact(str(exc), self.token)
        if status in _AUTH_STATUSES:
            return ProviderAuthenticationError(message)
        if status in _RATE_LIMIT_STATUSES:
            return ProviderRateLimitError(message)
        if status in _BAD_REQUEST_STATUSES:
            return ProviderBadRequestError(message)
        return ProviderUnavailableError(message)

    @staticmethod
    def _extract_text(output: Any) -> str:
        """Pull the first choice's message content out of any SDK response."""
        choices = getattr(output, "choices", None) or []
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        if message is None:
            return ""
        return getattr(message, "content", None) or ""


def _usage_to_dict(usage: Any) -> Optional[Dict[str, int]]:
    """Normalize token-usage objects (pydantic, dataclass, or dict) to a dict."""
    if usage is None:
        return None
    if isinstance(usage, dict):
        return dict(usage)
    if hasattr(usage, "model_dump"):  # pydantic v2
        return dict(usage.model_dump())
    try:
        return dict(asdict(usage))  # dataclass (ChatCompletionOutputUsage is one)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Capability run_fns -- the `capabilities/*.run` convention.
# ---------------------------------------------------------------------------


def _chat_run(
    text: str,
    instruction: Optional[str] = None,
    *,
    provider: HFProvider,
    system: Optional[str],
    temperature: float,
    max_tokens: int,
) -> str:
    """One capability execution: build messages, run, return plain text."""
    result = provider.execute(
        messages=_build_messages(text, instruction, system),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return result.text


def _asr_run(
    text: str,
    instruction: Optional[str] = None,
    *,
    provider: HFProvider,
    model: str,
) -> str:
    """Wired run_fn for automatic_speech_recognition (whisper models).

    `text` is treated as the path to an audio file. The HF Inference Client's
    `automatic_speech_recognition` method is called directly; the result text
    is the transcription.
    """
    if not provider.token:
        raise ProviderAuthenticationError(
            "Hugging Face ASR requires a token: set HF_TOKEN "
            "(or HUG) in the environment or .env"
        )
    audio_path = text.strip()
    from pathlib import Path as _P

    if not _P(audio_path).is_file():
        raise ProviderError(f"ASR input is not a valid file: {audio_path}")

    client = provider._client or provider._build_client()
    try:
        result = client.automatic_speech_recognition(
            model=model,
            audio=audio_path,
        )
    except (InferenceTimeoutError, TimeoutError, requests.exceptions.Timeout) as exc:
        raise ProviderTimeoutError(
            _redact(f"HF ASR timed out: {exc}", provider.token)
        ) from exc
    except (HfHubHTTPError, requests.HTTPError) as exc:
        raise provider._map_http_error(exc) from exc
    except requests.exceptions.RequestException as exc:
        raise ProviderUnavailableError(
            _redact(f"HF ASR network failure: {exc}", provider.token)
        ) from exc
    except ProviderError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ProviderError(
            _redact(f"HF ASR failed: {exc}", provider.token)
        ) from exc

    return getattr(result, "text", None) or str(result)


def _audio_chat_run(
    text: str,
    instruction: Optional[str] = None,
    *,
    provider: HFProvider,
    model: str,
    system: Optional[str],
    temperature: float,
    max_tokens: int,
) -> str:
    """Wired run_fn for audio_chat_completion (qwen2-audio models).

    `text` is treated as the path to an audio file. The audio is embedded in
    the chat message as an audio_url content part; the instruction (or a
    default) becomes the text part.
    """
    if not provider.token:
        raise ProviderAuthenticationError(
            "Hugging Face audio chat requires a token: set HF_TOKEN "
            "(or HUG) in the environment or .env"
        )
    audio_path = text.strip()
    from pathlib import Path as _P

    if not _P(audio_path).is_file():
        raise ProviderError(f"Audio chat input is not a valid file: {audio_path}")

    user_content = instruction or "Transcribe and summarize the audio."
    messages: List[Dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": {"url": audio_path}},
                {"type": "text", "text": user_content},
            ],
        }
    )

    client = provider._client or provider._build_client()
    chat_kwargs: Dict[str, Any] = {
        "messages": messages,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        output = client.chat_completion(**chat_kwargs)
    except (InferenceTimeoutError, TimeoutError, requests.exceptions.Timeout) as exc:
        raise ProviderTimeoutError(
            _redact(f"HF audio chat timed out: {exc}", provider.token)
        ) from exc
    except (HfHubHTTPError, requests.HTTPError) as exc:
        raise provider._map_http_error(exc) from exc
    except requests.exceptions.RequestException as exc:
        raise ProviderUnavailableError(
            _redact(f"HF audio chat network failure: {exc}", provider.token)
        ) from exc
    except ProviderError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ProviderError(
            _redact(f"HF audio chat failed: {exc}", provider.token)
        ) from exc

    return provider._extract_text(output)


def _audio_classification_run(
    text: str,
    instruction: Optional[str] = None,
    *,
    provider: HFProvider,
    model: str,
) -> str:
    """Wired run_fn for audio_classification (AST models).

    `text` is treated as the path to an audio file. Returns top-5 class
    labels with scores.
    """
    if not provider.token:
        raise ProviderAuthenticationError(
            "Hugging Face audio classification requires a token: set HF_TOKEN "
            "(or HUG) in the environment or .env"
        )
    audio_path = text.strip()
    from pathlib import Path as _P

    if not _P(audio_path).is_file():
        raise ProviderError(
            f"Audio classification input is not a valid file: {audio_path}"
        )

    client = provider._client or provider._build_client()
    try:
        results = client.audio_classification(
            model=model,
            audio=audio_path,
        )
    except (InferenceTimeoutError, TimeoutError, requests.exceptions.Timeout) as exc:
        raise ProviderTimeoutError(
            _redact(f"HF audio classification timed out: {exc}", provider.token)
        ) from exc
    except (HfHubHTTPError, requests.HTTPError) as exc:
        raise provider._map_http_error(exc) from exc
    except requests.exceptions.RequestException as exc:
        raise ProviderUnavailableError(
            _redact(f"HF audio classification network failure: {exc}", provider.token)
        ) from exc
    except ProviderError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ProviderError(
            _redact(f"HF audio classification failed: {exc}", provider.token)
        ) from exc

    if not results:
        return "(no classification results)"
    lines = []
    for r in results[:5]:
        label = getattr(r, "label", "?")
        score = getattr(r, "score", 0.0)
        lines.append(f"{label}: {score:.4f}")
    return "\n".join(lines)


def _build_messages(
    text: str, instruction: Optional[str], system: Optional[str]
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    user = text
    if instruction:
        user = f"{instruction}\n\n---\n\n{text}"
    messages.append({"role": "user", "content": user})
    return messages


def run_result(
    text: str,
    instruction: Optional[str] = None,
    *,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    system: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 2048,
    client: Any = None,
) -> HFResult:
    """Run one chat completion and return the full normalized `HFResult`.

    The evaluation layer uses this instead of `run()` so it can capture the
    metrics HF actually exposes (model, provider, token usage) in addition to
    the generated text. `run()` is the thin wrapper below that keeps the
    plain-str contract the registry / executor / FailureManager expect.
    """
    hf = HFProvider(model=model, provider=provider, client=client)
    return hf.execute(
        messages=_build_messages(text, instruction, system),
        temperature=temperature,
        max_tokens=max_tokens,
    )


def run(
    text: str,
    instruction: Optional[str] = None,
    *,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    system: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 2048,
    client: Any = None,
) -> str:
    """Compatibility wrapper matching the `capabilities/*.run(text, instruction)`
    signature used by the registry / executor / FailureManager."""
    return run_result(
        text,
        instruction,
        model=model,
        provider=provider,
        system=system,
        temperature=temperature,
        max_tokens=max_tokens,
        client=client,
    ).text


# ---------------------------------------------------------------------------
# Registry integration -- opt-in, keeps the default pool untouched.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HFModelSpec:
    """One Hugging Face model, declared as a Capability Registry resource.

    The entry *declares* what the model is represented as in the architecture:
    resource class, supported task, capability flags, io contract, execution
    interface and the configuration baked into its run_fn. It asserts no
    measured benchmark values -- cost/latency/availability use the
    architecture's declared defaults (see `build_hf_manifests`) until real
    telemetry exists.

    Every capability in `capabilities` MUST have an entry in
    `capability_provenance`: `"documented"` when the model card asserts it
    directly, `"inferred"` when it follows from documented properties (the
    registry report explains each inference). Anything that cannot be verified
    is left out of `capabilities` entirely -- unknown is recorded, never
    guessed.

    `interface` names the inference task the resource maps onto. Only
    `chat_completion` is wired to the adapter today; the other interfaces are
    declared for capability-based selection and raise a typed error if invoked.
    """

    resource_id: str
    model: str
    resource_class: str
    task: str
    capabilities: List[str]
    capability_provenance: Dict[str, str]
    input_type: str
    output_type: str
    output_format: str
    interface: str
    params: str
    context_length: str
    modality: str
    source: str
    price: str
    system: str = ""
    temperature: float = 0.2
    max_tokens: int = 2048


# Model catalog: ten real models chosen to give the registry enough capability
# diversity for capability-based selection to be meaningful (general text
# generation, reasoning, coding, instruction following, embeddings, reranking,
# vision-language, efficient small models, large high-capability models, and a
# specialized speech model). Capabilities and facts below come from the cited
# model cards / Hugging Face Inference Providers router metadata, fetched
# 2026-08-16. Prices are externally documented router prices, recorded in
# metadata only -- CostModel stays at the declared default (0.0) because the
# router's provider choice is "auto", so no single price is asserted for
# scoring.
HF_MODEL_CATALOG: List[HFModelSpec] = [
    HFModelSpec(
        resource_id="hf_qwen3_30b_a3b_instruct",
        model="Qwen/Qwen3-30B-A3B-Instruct",
        resource_class="llm",
        task="text-generation",
        capabilities=[
            "reasoning.deep",
            "reasoning.shallow",
            "tool.calling",
            "text.summarization",
            "text.classification",
            "answer.synthesis",
            "planning.decomposition",
        ],
        capability_provenance={
            # Card is gated (401). Reasoning + tool use asserted on the Qwen3
            # family card (Qwen/Qwen3-8B); the sibling Qwen/Qwen3-30B-A3B is
            # listed on the router with tools supported.
            "reasoning.deep": "documented",
            "reasoning.shallow": "documented",
            "tool.calling": "documented",
            "text.summarization": "inferred",
            "text.classification": "inferred",
            "answer.synthesis": "inferred",
            "planning.decomposition": "inferred",
        },
        input_type="text",
        output_type="text",
        output_format="plain",
        interface="chat_completion",
        params="~30B total / ~3.3B active (MoE, A3B); exact figures unverified (card gated)",
        context_length="40,960 (router/sibling); card gated",
        modality="text",
        source="https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct",
        price=(
            "sibling Qwen/Qwen3-30B-A3B served by deepinfra at $0.12/$0.50 "
            "per 1M tokens (router, 2026-08-16); exact-id price unverified"
        ),
        system=(
            "You are a precise, evidence-based assistant. Follow the user's "
            "instruction exactly, preserve all specific facts, numbers, dates "
            "and names, never invent details, and clearly flag anything "
            "unverifiable."
        ),
        temperature=0.2,
        max_tokens=2048,
    ),
    HFModelSpec(
        resource_id="hf_qwen3_8b",
        model="Qwen/Qwen3-8B",
        resource_class="llm",
        task="text-generation",
        capabilities=[
            "reasoning.shallow",
            "tool.calling",
            "text.summarization",
            "text.classification",
        ],
        capability_provenance={
            "reasoning.shallow": "documented",
            "tool.calling": "documented",
            "text.summarization": "inferred",
            "text.classification": "inferred",
        },
        input_type="text",
        output_type="text",
        output_format="plain",
        interface="chat_completion",
        params="8.2B total / 6.95B non-embedding (dense)",
        context_length="32,768 native / 131,072 via YaRN (router: 40,960)",
        modality="text",
        source="https://huggingface.co/Qwen/Qwen3-8B",
        price="nscale $0.07/$0.18 per 1M tokens (router, 2026-08-16)",
        system=(
            "You are a concise assistant. Follow the instruction, keep the "
            "output tight, preserve the key facts and numbers, and do not "
            "invent details."
        ),
        temperature=0.3,
        max_tokens=1024,
    ),
    HFModelSpec(
        resource_id="hf_qwen2_5_vl_7b_instruct",
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        resource_class="vlm",
        task="visual-question-answering",
        capabilities=[
            "vision.understanding",
            "vision_input",
            "image_understanding",
            "visual_question_answering",
            "visual_reasoning",
            "vision_language",
            "instruction_following",
            "tool.calling",
            "reasoning.shallow",
            "text.classification",
        ],
        capability_provenance={
            "vision.understanding": "documented",
            "vision_input": "documented",
            "image_understanding": "documented",
            "visual_question_answering": "documented",
            "visual_reasoning": "documented",
            "vision_language": "documented",
            "instruction_following": "inferred",
            "tool.calling": "documented",
            "reasoning.shallow": "documented",
            "text.classification": "inferred",
        },
        input_type="image",
        output_type="text",
        output_format="plain",
        interface="chat_completion",
        params="7B (dense)",
        context_length="32,768 (64K suggested for long video)",
        modality="text + image + video",
        source="https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct",
        price="not published (featherless-ai, live; billed per compute)",
        system=(
            "You are a vision-language assistant. Describe and interpret the "
            "image precisely; state uncertainty rather than guessing."
        ),
        temperature=0.2,
        max_tokens=1024,
    ),
    HFModelSpec(
        resource_id="hf_qwen3_coder_30b_a3b",
        model="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        resource_class="llm",
        task="text-generation",
        capabilities=[
            "code.generation",
            "tool.calling",
            "reasoning.deep",
            "reasoning.shallow",
        ],
        capability_provenance={
            "code.generation": "documented",
            "tool.calling": "documented",
            "reasoning.deep": "documented",
            "reasoning.shallow": "documented",
        },
        input_type="text",
        output_type="text",
        output_format="plain",
        interface="chat_completion",
        params="30.5B total / 3.3B active (MoE, 128 experts / 8 active)",
        context_length="262,144 native / 1M via YaRN",
        modality="text",
        source="https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct",
        price="scaleway $0.228/$0.912 per 1M tokens (router, 2026-08-16)",
        system=(
            "You are an expert software engineer. Solve the user's coding "
            "task precisely: write correct, idiomatic code, explain "
            "tradeoffs briefly, and never invent APIs or behavior that are "
            "not part of the task."
        ),
        temperature=0.2,
        max_tokens=4096,
    ),
    HFModelSpec(
        resource_id="hf_deepseek_r1_distill_qwen_32b",
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        resource_class="llm",
        task="text-generation",
        capabilities=["reasoning.deep", "reasoning.shallow"],
        capability_provenance={
            "reasoning.deep": "documented",
            "reasoning.shallow": "documented",
        },
        input_type="text",
        output_type="text",
        output_format="plain",
        interface="chat_completion",
        params="32B (dense; distilled from Qwen2.5-32B)",
        context_length="not stated on card",
        modality="text",
        source="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        price="not published (featherless-ai, live; billed per compute)",
        system=(
            "You are a careful reasoning assistant. Work through the problem "
            "step by step, show your reasoning, and give a precise final "
            "answer."
        ),
        temperature=0.1,
        max_tokens=4096,
    ),
    HFModelSpec(
        resource_id="hf_llama3_3_70b_instruct",
        model="meta-llama/Llama-3.3-70B-Instruct",
        resource_class="llm",
        task="text-generation",
        capabilities=[
            "reasoning.deep",
            "reasoning.shallow",
            "code.generation",
            "tool.calling",
            "text.summarization",
            "text.classification",
            "answer.synthesis",
            "planning.decomposition",
        ],
        capability_provenance={
            "reasoning.deep": "documented",
            "reasoning.shallow": "documented",
            "code.generation": "documented",
            "tool.calling": "documented",
            "text.summarization": "inferred",
            "text.classification": "inferred",
            "answer.synthesis": "inferred",
            "planning.decomposition": "inferred",
        },
        input_type="text",
        output_type="text",
        output_format="plain",
        interface="chat_completion",
        params="70B (dense; ~71B per HF metadata)",
        context_length="128K",
        modality="text in / text+code out (multilingual)",
        source="https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct",
        price=(
            "novita $0.135/$0.40 per 1M tokens (cheapest live router "
            "provider, 2026-08-16; also groq $0.59/$0.79, together $1.04/$1.04)"
        ),
        system=(
            "You are a precise, evidence-based assistant. Follow the user's "
            "instruction exactly, preserve all specific facts, numbers, dates "
            "and names, never invent details, and clearly flag anything "
            "unverifiable."
        ),
        temperature=0.2,
        max_tokens=4096,
    ),
    HFModelSpec(
        resource_id="hf_llama3_2_3b_instruct",
        model="meta-llama/Llama-3.2-3B-Instruct",
        resource_class="llm",
        task="text-generation",
        capabilities=[
            "text.summarization",
            "tool.calling",
            "reasoning.shallow",
            "text.classification",
        ],
        capability_provenance={
            "text.summarization": "documented",
            "tool.calling": "documented",
            "reasoning.shallow": "documented",
            "text.classification": "inferred",
        },
        input_type="text",
        output_type="text",
        output_format="plain",
        interface="chat_completion",
        params="3B (dense; 3.21B)",
        context_length="128K",
        modality="text in / text+code out (multilingual)",
        source="https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct",
        price="not published (featherless-ai, live; billed per compute)",
        system=(
            "You are a concise assistant. Follow the instruction, keep the "
            "output tight, preserve the key facts and numbers, and do not "
            "invent details."
        ),
        temperature=0.3,
        max_tokens=1024,
    ),
    HFModelSpec(
        resource_id="hf_gemma2_27b_it",
        model="google/gemma-2-27b-it",
        resource_class="llm",
        task="text-generation",
        capabilities=[
            "reasoning.deep",
            "reasoning.shallow",
            "text.summarization",
            "code.generation",
            "text.classification",
        ],
        capability_provenance={
            "reasoning.deep": "documented",
            "reasoning.shallow": "documented",
            "text.summarization": "documented",
            "code.generation": "documented",
            "text.classification": "inferred",
        },
        input_type="text",
        output_type="text",
        output_format="plain",
        interface="chat_completion",
        params="27B (dense)",
        context_length="not stated on card",
        modality="text (English)",
        source="https://huggingface.co/google/gemma-2-27b-it",
        price="not published (featherless-ai, live; billed per compute)",
        system=(
            "You are a helpful assistant. Follow the instruction, preserve "
            "specific facts and numbers, and never invent details."
        ),
        temperature=0.2,
        max_tokens=2048,
    ),
    HFModelSpec(
        resource_id="hf_bge_large_en_v1_5",
        model="BAAI/bge-large-en-v1.5",
        resource_class="embedder",
        task="feature-extraction",
        capabilities=["embedding.generation"],
        capability_provenance={"embedding.generation": "documented"},
        input_type="text",
        output_type="text",
        output_format="plain",
        interface="feature_extraction",
        params="335M (dense BERT bi-encoder; 1024-dim embeddings)",
        context_length="512 tokens (sequence length, MTEB)",
        modality="text (sentence embeddings; English)",
        source="https://huggingface.co/BAAI/bge-large-en-v1.5",
        price="not published (hf-inference, feature-extraction; billed by compute time)",
        system="",
        temperature=0.0,
        max_tokens=0,
    ),
    HFModelSpec(
        resource_id="hf_bge_reranker_v2_m3",
        model="BAAI/bge-reranker-v2-m3",
        resource_class="reranker",
        task="text-reranking",
        capabilities=["rerank.scoring"],
        capability_provenance={"rerank.scoring": "documented"},
        input_type="text",
        output_type="text",
        output_format="plain",
        interface="rerank",
        params="568M (dense cross-encoder; based on bge-m3)",
        context_length="512 tokens (usage examples truncate at max_length=512)",
        modality="text (multilingual reranking)",
        source="https://huggingface.co/BAAI/bge-reranker-v2-m3",
        price="not published (hf-inference, text-classification; billed by compute time)",
        system="",
        temperature=0.0,
        max_tokens=0,
    ),
    HFModelSpec(
        resource_id="hf_whisper_large_v3_turbo",
        model="openai/whisper-large-v3-turbo",
        resource_class="asr",
        task="automatic-speech-recognition",
        capabilities=[
            "audio_input",
            "speech_recognition",
            "automatic_speech_recognition",
            "speech_to_text",
            "transcription",
            "multilingual_speech",
            "speech.transcription",
        ],
        capability_provenance={
            "audio_input": "documented",
            "speech_recognition": "documented",
            "automatic_speech_recognition": "documented",
            "speech_to_text": "inferred",
            "transcription": "documented",
            "multilingual_speech": "documented",
            "speech.transcription": "documented",
        },
        input_type="audio",
        output_type="text",
        output_format="plain",
        interface="automatic_speech_recognition",
        params="809M (dense encoder-decoder)",
        context_length="30-second audio receptive field (token context not stated)",
        modality="audio (99 languages)",
        source="https://huggingface.co/openai/whisper-large-v3-turbo",
        price="not published (hf-inference/deepinfra; billed by compute time)",
        system="",
        temperature=0.0,
        max_tokens=0,
    ),
    # ------------------------------------------------------------------
    # Multimodal / specialized models (2026-08-16 sources)
    # ------------------------------------------------------------------
    # OpenAI Whisper Large V3 -- full-scale ASR (1.55B, 99 languages).
    HFModelSpec(
        resource_id="hf_whisper_large_v3",
        model="openai/whisper-large-v3",
        resource_class="asr",
        task="automatic-speech-recognition",
        capabilities=[
            "audio_input",
            "speech_recognition",
            "automatic_speech_recognition",
            "speech_to_text",
            "transcription",
            "multilingual_speech",
            "speech.transcription",
        ],
        capability_provenance={
            "audio_input": "documented",
            "speech_recognition": "documented",
            "automatic_speech_recognition": "documented",
            "speech_to_text": "inferred",
            "transcription": "documented",
            "multilingual_speech": "documented",
            "speech.transcription": "documented",
        },
        input_type="audio",
        output_type="text",
        output_format="plain",
        interface="automatic_speech_recognition",
        params="1.55B (dense encoder-decoder; 1.54B weights)",
        context_length="30-second receptive field (optimal chunk 30s; long-form via chunking)",
        modality="audio (99 languages) -> text",
        source="https://huggingface.co/openai/whisper-large-v3",
        price=(
            "fal + others on HF router (2026-08-16); "
            "audio transport not wired; billed by compute time, not tokens"
        ),
        system="",
        temperature=0.0,
        max_tokens=0,
    ),
    # Qwen2-Audio 7B -- audio-language chat model.
    HFModelSpec(
        resource_id="hf_qwen2_audio_7b_instruct",
        model="Qwen/Qwen2-Audio-7B-Instruct",
        resource_class="audio",
        task="audio-text-to-text",
        capabilities=[
            "audio_input",
            "audio_understanding",
            "speech_understanding",
            "audio_analysis",
            "audio_to_text",
            "instruction_following",
        ],
        capability_provenance={
            "audio_input": "documented",
            "audio_understanding": "inferred",
            "speech_understanding": "inferred",
            "audio_analysis": "documented",
            "audio_to_text": "inferred",
            "instruction_following": "inferred",
        },
        input_type="audio",
        output_type="text",
        output_format="plain",
        interface="audio_chat_completion",
        params="8.4B total (audio encoder + Qwen2 text decoder)",
        context_length="not stated on card",
        modality="audio + text -> text",
        source="https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct",
        price="not deployed by any Inference Provider (2026-08-16); audio transport not wired",
        system="",
        temperature=0.0,
        max_tokens=0,
    ),
    # MIT AST -- AudioSet sound/event classification.
    HFModelSpec(
        resource_id="hf_ast_audioset_finetuned",
        model="MIT/ast-finetuned-audioset-10-10-0.4593",
        resource_class="audio",
        task="audio-classification",
        capabilities=[
            "audio_input",
            "audio_classification",
            "sound_classification",
            "audio_event_recognition",
        ],
        capability_provenance={
            "audio_input": "documented",
            "audio_classification": "documented",
            "sound_classification": "inferred",
            "audio_event_recognition": "inferred",
        },
        input_type="audio",
        output_type="text",
        output_format="plain",
        interface="audio_classification",
        params="86.6M (Audio Spectrogram Transformer finetuned on AudioSet)",
        context_length="~10-second spectrogram clips (implied by checkpoint id audioset-10-10)",
        modality="audio -> class labels (AudioSet 527 classes)",
        source="https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593",
        price="not deployed by any Inference Provider (2026-08-16); audio transport not wired",
        system="",
        temperature=0.0,
        max_tokens=0,
    ),
    # SigLIP 2 -- zero-shot classification, image-text matching, vision encoder.
    HFModelSpec(
        resource_id="hf_siglip2_base_224",
        model="google/siglip2-base-patch16-224",
        resource_class="image",
        task="zero-shot-image-classification",
        capabilities=[
            "vision_input",
            "image_classification",
            "zero_shot_classification",
            "image_text_matching",
            "image_text_retrieval",
            "visual_feature_extraction",
        ],
        capability_provenance={
            "vision_input": "documented",
            "image_classification": "inferred",
            "zero_shot_classification": "documented",
            "image_text_matching": "inferred",
            "image_text_retrieval": "documented",
            "visual_feature_extraction": "documented",
        },
        input_type="image",
        output_type="text",
        output_format="plain",
        interface="zero_shot_image_classification",
        params="~0.4B total (ViT-B/16 image tower + text tower; 375M weights)",
        context_length="224x224 input (patch16; aspect-ratio and resolution adaptibility in training)",
        modality="image + text labels -> classification/similarity scores, image & text embeddings",
        source="https://huggingface.co/google/siglip2-base-patch16-224",
        price="not deployed by any Inference Provider (2026-08-16); vision transport not wired",
        system="",
        temperature=0.0,
        max_tokens=0,
    ),
    # Ultralytics YOLO11 -- object detection (weight archive, not transformers).
    HFModelSpec(
        resource_id="hf_yolo11",
        model="Ultralytics/YOLO11",
        resource_class="image",
        task="object-detection",
        capabilities=[
            "vision_input",
            "object_detection",
            "object_identification",
            "object_localization",
            "multi_object_detection",
        ],
        capability_provenance={
            "vision_input": "documented",
            "object_detection": "documented",
            "object_identification": "inferred",
            "object_localization": "inferred",
            "multi_object_detection": "inferred",
        },
        input_type="image",
        output_type="structured",
        output_format="json",
        interface="object_detection",
        params=(
            "2.6M (yolo11n) - 56.9M (yolo11x) detection variants "
            "(.pt weight archive, not a transformers pipeline model)"
        ),
        context_length="640x640 input (detection; 224 classification, 1024 OBB per card)",
        modality="image -> bounding boxes, class labels, confidence scores",
        source="https://huggingface.co/Ultralytics/YOLO11",
        price=(
            "not deployed by any Inference Provider (2026-08-16); "
            ".pt weight archive, not a serverless endpoint"
        ),
        system="",
        temperature=0.0,
        max_tokens=0,
    ),
    # Facebook DINOv3 ViT-B/16 -- self-supervised vision backbone / embeddings.
    HFModelSpec(
        resource_id="hf_dinov3_vitb16",
        model="facebook/dinov3-vitb16-pretrain-lvd1689m",
        resource_class="image",
        task="image-feature-extraction",
        capabilities=[
            "vision_input",
            "visual_feature_extraction",
            "image_representation",
            "visual_embedding",
            "image_similarity",
        ],
        capability_provenance={
            "vision_input": "documented",
            "visual_feature_extraction": "documented",
            "image_representation": "documented",
            "visual_embedding": "documented",
            "image_similarity": "inferred",
        },
        input_type="image",
        output_type="structured",
        output_format="json",
        interface="image_feature_extraction",
        params="86M (ViT-B/16; 85.7M weights)",
        context_length=(
            "224x224 -> 1 class + 4 register + 196 patch tokens = 201 tokens; "
            "larger images OK if multiples of patch size 16"
        ),
        modality="image -> class / patch / register embedding vectors",
        source="https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m",
        price=(
            "not deployed by any Inference Provider (2026-08-16); "
            "model card gated (manual Meta approval form)"
        ),
        system="",
        temperature=0.0,
        max_tokens=0,
    ),
]


def build_hf_manifests(
    *, provider: Optional[str] = None
) -> List["CapabilityManifest"]:
    """Build one CapabilityManifest per catalog model (DOC1 5.1 declarations).

    Each model is registered as a *resource* in its own right, exposing the
    capability flags it is declared to support. Scheduling binds on
    `capabilities` alone; `metadata` carries provider/model/task/interface and
    the model-card provenance (params, context length, modality, source,
    externally documented price, per-capability documented/inferred markers)
    and is trace data, never decision logic.

    Cost, latency, availability and quality are deliberately NOT asserted here:
    there are no measured benchmark numbers, so every entry uses the
    architecture's declared defaults (CostModel/LatencyModel/Availability
    defaults and empty quality_priors -> neutral 0.5). The externally
    documented router price is recorded in `metadata.price` only -- the router
    routes to whichever provider it picks, so asserting one price in the
    CostModel would pretend a precision the auto-router does not have. Real
    telemetry can later replace those defaults without touching any routing
    code.

    Note: the adapter's transport is text chat. Chat-completion models get
    `transport="wired"`; the embedding / rerank / ASR entries declare their
    interface (`transport="declared"`) and their run_fn raises a typed error
    if invoked -- image and audio payload support are declared interfaces, not
    yet implemented (mirrors the vision model's existing note).
    """
    import json as _json

    from capability_registry import (
        Availability,
        CapabilityManifest,
        CostModel,
        IOSchema,
        LatencyModel,
    )
    from config import HF_PROVIDER as _CFG_PROVIDER

    hf_provider_sel = provider or _CFG_PROVIDER

    manifests = []
    for spec in HF_MODEL_CATALOG:
        _validate_spec(spec)
        _wired_interfaces = {
            "chat_completion",
            "automatic_speech_recognition",
            "audio_chat_completion",
            "audio_classification",
        }
        transport = "wired" if spec.interface in _wired_interfaces else "declared"
        manifests.append(
            CapabilityManifest(
                resource_id=spec.resource_id,
                resource_class=spec.resource_class,
                capabilities=list(spec.capabilities),
                input_schema=IOSchema(type=spec.input_type, format="plain"),
                output_schema=IOSchema(
                    type=spec.output_type, format=spec.output_format
                ),
                cost_model=CostModel(unit="per_1k_tokens", estimate_usd=0.0),
                latency_model=LatencyModel(),
                availability=Availability(),
                risk_class="low",
                metadata={
                    "provider": "huggingface",
                    "hf_provider": hf_provider_sel,
                    "model": spec.model,
                    "task": spec.task,
                    "interface": spec.interface,
                    "transport": transport,
                    "params": spec.params,
                    "context_length": spec.context_length,
                    "modality": spec.modality,
                    "source": spec.source,
                    "price": spec.price,
                    "capability_provenance": _json.dumps(
                        spec.capability_provenance, sort_keys=True
                    ),
                    "declared": (
                        "capabilities declared from model documentation; "
                        "cost/latency/quality unmeasured (architecture defaults)"
                    ),
                },
            )
        )
    return manifests


def _validate_spec(spec: "HFModelSpec") -> None:
    """Fail fast on a malformed catalog entry at build time, not at runtime."""
    if set(spec.capabilities) != set(spec.capability_provenance):
        raise ValueError(
            f"[model-catalog] {spec.resource_id}: capabilities {set(spec.capabilities)} "
            f"must exactly match capability_provenance keys "
            f"{set(spec.capability_provenance)}"
        )
    unknown_prov = {
        f: p
        for f, p in spec.capability_provenance.items()
        if p not in ("documented", "inferred")
    }
    if unknown_prov:
        raise ValueError(
            f"[model-catalog] {spec.resource_id}: provenance must be "
            f"'documented' or 'inferred', got {unknown_prov}"
        )


def _declared_only_run(resource_id: str, interface: str):
    """run_fn for resources whose transport is not wired to the text pipeline.

    These resources exist to prove capability-based *selection* (the registry
    layer), not execution: invoking them raises a typed, explicit error rather
    than silently returning garbage.
    """

    def run(text, instruction=None):
        raise ProviderError(
            f"[capability-registry] resource '{resource_id}' declares interface "
            f"'{interface}' which is not wired to the text pipeline; registered "
            f"for capability-based selection, not execution"
        )

    return run


def register_hf_resources(
    registry: "CapabilityRegistry",
    *,
    provider: Optional[str] = None,
    client: Any = None,
) -> List["CapabilityManifest"]:
    """Register one resource per Hugging Face catalog model (opt-in).

    The default pool (`build_default_registry`) is left alone -- call this only
    when a run explicitly wants the Hugging Face provider in the candidate set.
    Chat-completion resources are bound to the HF adapter pinned to that model;
    ASR resources use `automatic_speech_recognition`; audio-chat resources use
    `chat_completion` with audio content parts; audio-classification resources
    use `audio_classification`. Remaining declared-only resources (embedding /
    rerank) get a run_fn that raises a typed error if invoked. Returns the
    registered manifests.
    """
    manifests = build_hf_manifests(provider=provider)

    run_fns = {}
    for spec in HF_MODEL_CATALOG:
        hf = HFProvider(model=spec.model, provider=provider, client=client)

        if spec.interface == "chat_completion":
            run_fns[spec.resource_id] = partial(
                _chat_run,
                provider=hf,
                system=spec.system,
                temperature=spec.temperature,
                max_tokens=spec.max_tokens,
            )
        elif spec.interface == "automatic_speech_recognition":
            run_fns[spec.resource_id] = partial(
                _asr_run,
                provider=hf,
                model=spec.model,
            )
        elif spec.interface == "audio_chat_completion":
            run_fns[spec.resource_id] = partial(
                _audio_chat_run,
                provider=hf,
                model=spec.model,
                system=spec.system,
                temperature=spec.temperature,
                max_tokens=spec.max_tokens,
            )
        elif spec.interface == "audio_classification":
            run_fns[spec.resource_id] = partial(
                _audio_classification_run,
                provider=hf,
                model=spec.model,
            )
        else:
            run_fns[spec.resource_id] = _declared_only_run(
                spec.resource_id, spec.interface
            )

    for manifest in manifests:
        registry.register(manifest, run_fns[manifest.resource_id])
    return manifests

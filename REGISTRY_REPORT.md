# Registry Report — Expanded Model Catalog

Generated from the live catalog in `providers/hf.py::HF_MODEL_CATALOG`
(sources fetched 2026-08-16). Machine-readable mirror:
`outputs/registry_data.json` (regenerate with `python registry_export.py`).

## Scope

The default pool (`build_default_registry()`) is untouched — the DOC2 control
group stays frozen. The HF pool is **opt-in** and registers **17 real models**
across **15 capability classes** as ordinary `CapabilityManifest` resources.
Nothing in the core (`models.py` scheduling, agents, executor) references a
model or provider by name; routing happens on capability flags alone.

Two additive schema extensions were required to express the catalog honestly:

| Change | Where |
|---|---|
| `CAPABILITY_FLAGS` += `embedding.generation`, `rerank.scoring`, `audio_input`, `speech_recognition`, `automatic_speech_recognition`, `speech_to_text`, `transcription`, `multilingual_speech`, `audio_understanding`, `speech_understanding`, `audio_analysis`, `audio_to_text`, `instruction_following`, `audio_classification`, `sound_classification`, `audio_event_recognition`, `vision_input`, `image_classification`, `zero_shot_classification`, `image_text_matching`, `image_text_retrieval`, `visual_feature_extraction`, `object_detection`, `object_identification`, `object_localization`, `multi_object_detection`, `image_understanding`, `visual_question_answering`, `visual_reasoning`, `vision_language`, `image_representation`, `visual_embedding`, `image_similarity` | `models.py` |
| `ResourceClass` += `embedder`, `reranker`, `audio`, `image` | `capability_registry.py` |

## Models

`providers/hf.py` — 8 chat-completion models wired to the HF adapter + 9
declared-only interfaces (embedding, reranking, 2 ASR, audio understanding,
audio classification, image classification, object detection, image feature
extraction). Every capability flag carries a provenance marker: **documented** =
asserted on the model card, **inferred** = implied by documented properties
(rationale below).

| resource_id | Model | Provider | Primary task | Capabilities (d/i) | Modality | Constraints | Source |
|---|---|---|---|---|---|---|---|
| `hf_qwen3_30b_a3b_instruct` | Qwen/Qwen3-30B-A3B-Instruct | HF (auto) | text-generation | reasoning.deep (d), reasoning.shallow (d), tool.calling (d), text.summarization (i), text.classification (i), answer.synthesis (i), planning.decomposition (i) | text | ~30B total / ~3.3B active MoE; ctx 40,960 (sibling/router); card gated | https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct |
| `hf_qwen3_8b` | Qwen/Qwen3-8B | HF (auto) | text-generation | reasoning.shallow (d), tool.calling (d), text.summarization (i), text.classification (i) | text | 8.2B dense; ctx 32,768 native / 131K YaRN | https://huggingface.co/Qwen/Qwen3-8B |
| `hf_qwen2_5_vl_7b_instruct` | Qwen/Qwen2.5-VL-7B-Instruct | HF (auto) | visual-question-answering | vision.understanding (d), vision_input (d), image_understanding (d), visual_question_answering (d), visual_reasoning (d), vision_language (d), instruction_following (i), tool.calling (d), reasoning.shallow (d), text.classification (i) | text + image + video | 7B dense; ctx 32,768 (64K suggested for long video) | https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct |
| `hf_qwen3_coder_30b_a3b` | Qwen/Qwen3-Coder-30B-A3B-Instruct | HF (auto) | text-generation (coding) | code.generation (d), tool.calling (d), reasoning.deep (d), reasoning.shallow (d) | text | 30.5B total / 3.3B active MoE (128/8); ctx 262,144 native / 1M YaRN | https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct |
| `hf_deepseek_r1_distill_qwen_32b` | deepseek-ai/DeepSeek-R1-Distill-Qwen-32B | HF (auto) | text-generation (reasoning) | reasoning.deep (d), reasoning.shallow (d) | text | 32B dense (distilled from Qwen2.5-32B); ctx not stated | https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B |
| `hf_llama3_3_70b_instruct` | meta-llama/Llama-3.3-70B-Instruct | HF (auto) | text-generation | reasoning.deep (d), reasoning.shallow (d), code.generation (d), tool.calling (d), text.summarization (i), text.classification (i), answer.synthesis (i), planning.decomposition (i) | text in / text+code out | 70B dense; ctx 128K | https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct |
| `hf_llama3_2_3b_instruct` | meta-llama/Llama-3.2-3B-Instruct | HF (auto) | text-generation (small) | text.summarization (d), tool.calling (d), reasoning.shallow (d), text.classification (i) | text in / text+code out | 3.21B dense; ctx 128K | https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct |
| `hf_gemma2_27b_it` | google/gemma-2-27b-it | HF (auto) | text-generation | reasoning.deep (d), reasoning.shallow (d), text.summarization (d), code.generation (d), text.classification (i) | text (English) | 27B dense; ctx not stated | https://huggingface.co/google/gemma-2-27b-it |
| `hf_bge_large_en_v1_5` | BAAI/bge-large-en-v1.5 | HF (auto) | feature-extraction | embedding.generation (d) | text (sentence embeddings, English) | 335M, 1024-dim; seq len 512 | https://huggingface.co/BAAI/bge-large-en-v1.5 |
| `hf_bge_reranker_v2_m3` | BAAI/bge-reranker-v2-m3 | HF (auto) | text-reranking | rerank.scoring (d) | text (multilingual reranking) | 568M cross-encoder (bge-m3 based); max len 512 | https://huggingface.co/BAAI/bge-reranker-v2-m3 |
| `hf_whisper_large_v3_turbo` | openai/whisper-large-v3-turbo | HF (auto) | ASR | audio_input (d), speech_recognition (d), automatic_speech_recognition (d), speech_to_text (i), transcription (d), multilingual_speech (d), speech.transcription (d) | audio (99 languages) | 809M; 30 s audio receptive field | https://huggingface.co/openai/whisper-large-v3-turbo |
| `hf_whisper_large_v3` | openai/whisper-large-v3 | HF (auto) | ASR | audio_input (d), speech_recognition (d), automatic_speech_recognition (d), speech_to_text (i), transcription (d), multilingual_speech (d), speech.transcription (d) | audio (99 languages) | 1.55B; 30 s audio receptive field | https://huggingface.co/openai/whisper-large-v3 |
| `hf_qwen2_audio_7b_instruct` | Qwen/Qwen2-Audio-7B-Instruct | HF (auto) | audio-text-to-text | audio_input (d), audio_understanding (i), speech_understanding (i), audio_analysis (d), audio_to_text (i), instruction_following (i) | audio + text → text | 8.4B (audio encoder + Qwen2 decoder); ctx not stated | https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct |
| `hf_ast_audioset_finetuned` | MIT/ast-finetuned-audioset-10-10-0.4593 | HF (auto) | audio-classification | audio_input (d), audio_classification (d), sound_classification (i), audio_event_recognition (i) | audio → class labels | 86.6M; ~10 s spectrogram clips | https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593 |
| `hf_siglip2_base_224` | google/siglip2-base-patch16-224 | HF (auto) | zero-shot-image-classification | vision_input (d), image_classification (i), zero_shot_classification (d), image_text_matching (i), image_text_retrieval (d), visual_feature_extraction (d) | image + text labels → scores/embeddings | ~0.4B; 224x224 input (patch16) | https://huggingface.co/google/siglip2-base-patch16-224 |
| `hf_yolo11` | Ultralytics/YOLO11 | HF (auto) | object-detection | vision_input (d), object_detection (d), object_identification (i), object_localization (i), multi_object_detection (i) | image → boxes/labels | 2.6M–56.9M (.pt archive, not transformers) | https://huggingface.co/Ultralytics/YOLO11 |
| `hf_dinov3_vitb16` | facebook/dinov3-vitb16-pretrain-lvd1689m | HF (auto) | image-feature-extraction | vision_input (d), visual_feature_extraction (d), image_representation (d), visual_embedding (d), image_similarity (i) | image → embedding vectors | 86M (ViT-B/16); 224x224 → 201 tokens | https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m |

## Capability profiles covered

The 17 models cover the target profiles:

1. **General text generation** — all 8 chat models
2. **Reasoning (deep)** — Qwen3-30B-A3B, Qwen3-Coder, DeepSeek-R1-Distill-32B, Llama-3.3-70B, Gemma-2-27B
3. **Coding** — Qwen3-Coder, Llama-3.3-70B, Gemma-2-27B
4. **Instruction following / tool calling** — Qwen3-30B-A3B, Qwen3-8B, Qwen2.5-VL, Qwen3-Coder, Llama-3.3-70B, Llama-3.2-3B
5. **Embeddings** — BGE-large-en-v1.5
6. **Reranking** — BGE-reranker-v2-M3
7. **Vision-language** — Qwen2.5-VL-7B
8. **Small / efficient** — Llama-3.2-3B, Qwen3-8B, BGE-large (335M), BGE-reranker (568M), Whisper-turbo (809M), Whisper-large-v3 (1.55B), AST (86.6M), SigLIP2 (~0.4B), YOLO11n (2.6M), DINOv3 (86M), plus the two 3.3B-active MoE models
9. **Large / high-capability** — Llama-3.3-70B, Qwen3-30B-A3B, Qwen3-Coder, Qwen2-Audio (8.4B), DeepSeek-32B, Gemma-2-27B
10. **Speech transcription / ASR** — Whisper-large-v3-turbo, Whisper-large-v3
11. **Audio understanding** — Qwen2-Audio-7B
12. **Audio / sound classification** — AST-AudioSet
13. **Image classification / zero-shot** — SigLIP2
14. **Object detection** — YOLO11
15. **Visual representation / embedding** — DINOv3 ViT-B/16

## Provenance (documented vs inferred) — explanations

**Documented** flags are asserted on the model card or on the authoritative
family card (see *Assumptions* for the gated card). **Inferred** flags come
from the following reasoning, never from benchmark claims:

| Inferred flag | Why |
|---|---|
| text.summarization on Qwen3-30B-A3B, Qwen3-8B | General instruction-following chat models; summarization is a standard covered task, but no card statement was found. |
| text.classification on Qwen3-30B-A3B, Qwen3-8B, Qwen2.5-VL, Llama-3.3, Llama-3.2-3B, Gemma-2 | All are instruct models that can label/sort text; treated as inferred rather than asserted because the cards emphasize chat/coding, not classification. |
| answer.synthesis on Qwen3-30B-A3B, Llama-3.3 | Aggregation/answer-generation is the documented use of the Qwen3/Llama instruct line, but it is a workflow claim, not a card assertion. |
| planning.decomposition on Qwen3-30B-A3B, Llama-3.3 | Follows from documented multi-step reasoning + tool use; not separately asserted. |
| instruction_following on Qwen2.5-VL, Qwen2-Audio | Both are chat/instruct models that follow user instructions; the phrase "instruction following" never appears on the cards, but both models explicitly accept and respond to typed instructions. |
| speech_to_text on both Whisper models | Cards say "transcribe" / "speech transcription" but never the literal phrase "speech to text". |
| audio_understanding, speech_understanding on Qwen2-Audio | Card says "audio analysis" and "speech instructions"; "understanding" is a close paraphrase, not a card assertion. |
| audio_to_text on Qwen2-Audio | The model's output is text responses to audio inputs, but the phrase "audio to text" is not used. |
| sound_classification, audio_event_recognition on AST | AudioSet classes are audio events, but the card only says "audio" (never "sound" or "event"). |
| image_classification, image_text_matching on SigLIP2 | Zero-shot classification is performed via image–text contrastive matching, but neither phrase appears as an explicit task name. |
| object_identification, object_localization, multi_object_detection on YOLO11 | mAP metrics and bounding-box outputs imply these, but the card only asserts "object detection" directly. |
| image_similarity on DINOv3 | "Image retrieval using nearest neighbors" implies similarity, but the word "similarity" is not used. |

Every catalog model carries **at least one documented** capability, and
`_validate_spec` enforces `capabilities == provenance keys` and provenance
values ∈ {documented, inferred} at manifest-build time.

## Numbers discipline (no fabricated scores)

- **Cost**: `CostModel` = 0.0 (`per_1k_tokens`) for every HF resource. Externally
  documented router prices are recorded in `metadata.price` **only**, because the
  router provider is `auto` and asserting one price would overclaim precision.
- **Latency**: declared defaults (p50 1000 ms / p95 4000 ms).
- **Quality**: empty `quality_priors` → neutral 0.5.
- **Availability**: declared default (`up`).
- Result: all HF resources are feasible under any cost ceiling ≥ 0 and any SLO ≥ 4 s;
  selection among them is a deterministic tie-break by `resource_id`.
- The **default pool keeps its measured/declared priors** (e.g. `vision.understanding`
  0.88), so in the mixed registry vision still binds the default resource.

Router prices recorded (per 1M tokens, 2026-08-16): Qwen3-8B nscale
$0.07/$0.18; Qwen3-Coder-30B-A3B scaleway $0.228/$0.912; Llama-3.3-70B novita
$0.135/$0.40 (also groq $0.59/$0.79, together $1.04/$1.04); all others "not
published (billed per compute time)".

## Transports

| Interface | resource_id | transport | run_fn |
|---|---|---|---|
| chat_completion | 8 chat models (qwen3-30b, qwen3-8b, vlm, coder, deepseek, llama3.3, llama3.2, gemma2) | `wired` | calls the HF adapter with the pinned model |
| feature_extraction | `hf_bge_large_en_v1_5` | `declared` | raises `ProviderError` naming the interface |
| rerank | `hf_bge_reranker_v2_m3` | `declared` | raises `ProviderError` naming the interface |
| automatic_speech_recognition | `hf_whisper_large_v3_turbo`, `hf_whisper_large_v3` | `declared` | raises `ProviderError` naming the interface |
| audio_chat_completion | `hf_qwen2_audio_7b_instruct` | `declared` | raises `ProviderError` naming the interface |
| audio_classification | `hf_ast_audioset_finetuned` | `declared` | raises `ProviderError` naming the interface |
| zero_shot_image_classification | `hf_siglip2_base_224` | `declared` | raises `ProviderError` naming the interface |
| object_detection | `hf_yolo11` | `declared` | raises `ProviderError` naming the interface |
| image_feature_extraction | `hf_dinov3_vitb16` | `declared` | raises `ProviderError` naming the interface |

The declared-not-wired pattern is deliberate: `huggingface_hub==0.35.0`'s
`InferenceClient` exposes `feature_extraction`, `rerank`-adjacent
text-classification and ASR endpoints, but the adapter's text transport is chat;
registering the interface now with an explicit error keeps the capability
queryable while never faking execution (mirrors the VLM's image-input note).

## Querying

```python
from capability_registry import CapabilityRegistry
from models import CapabilityDNA, DNAConstraints
import providers.hf as hf_mod

reg = CapabilityRegistry()
hf_mod.register_hf_resources(reg)

reg.find(["reasoning.deep", "code.generation"])     # -> coding agents
reg.find(["embedding.generation"])                  # -> bge-large-en-v1.5
reg.find(["web.search"])                            # -> []  (no-match, not guessed)

dna = CapabilityDNA(flags=["code.generation", "tool.calling", "reasoning.deep"],
                    constraints=DNAConstraints(cost_ceiling_usd=0.05, latency_slo_ms=30_000))
reg.select(dna).resource_id                          # -> hf_llama3_3_70b_instruct
```

Efficient-inference query: filter `metadata.params` for small/active parameter
counts — `test_model_catalog.py::_efficient_resources` derives the set from
documented params (e.g. Llama-3.2-3B 3.2B, Qwen3-8B 8.2B, BGE/Whisper sub-1B,
and the two 3.3B-active MoE models).

## Assumptions and unverifiable capabilities

- **Qwen/Qwen3-30B-A3B-Instruct card is gated (HTTP 401)** — its own card could
  not be read. Evidence used instead: (a) the Qwen3 family card (Qwen/Qwen3-8B,
  un-gated) asserts hybrid thinking/reasoning and tool use for the family;
  (b) the sibling `Qwen/Qwen3-30B-A3B` is listed on the HF router with tools
  supported at 40,960 context. Params "~30B/3.3B active" and context 40,960 are
  therefore **unverified for this exact id** and flagged as such in `params` and
  `context_length`.
- **facebook/dinov3-vitb16-pretrain-lvd1689m is gated (manual approval form)** —
  card text is readable but weights require Meta's approval form. Registered
  for capability-based selection, not execution.
- **Ultralytics/YOLO11 is a weight archive (.pt files), not a transformers
  pipeline model** — registered for capability-based selection; the HF
  auto-generated code snippet warns "Couldn't find a valid YOLO version tag."
  Declared-only; not executable via HF Inference Providers.
- **Qwen/Qwen2-Audio-7B-Instruct is not deployed by any Inference Provider**
  as of 2026-08-16 — registered for capability-based selection; audio transport
  not wired.
- **MIT/ast-finetuned-audioset-10-10-0.4593 is not deployed by any Inference
  Provider** — registered for capability-based selection; audio transport not
  wired.
- **google/siglip2-base-patch16-224 is not deployed by any Inference Provider**
  — registered for capability-based selection; vision transport not wired.
- Context length for DeepSeek-R1-Distill-Qwen-32B and Gemma-2-27B-it is **not
  stated on the card** — recorded as `not stated` rather than guessed.
- Prices are router listings (per-token, 2026-08-16), not a billing contract;
  several are billed per compute time and marked "not published".
- No benchmark/quality/latency numbers exist in this repo for these models; the
  registry deliberately stays silent on them (neutral priors) instead of
  inserting made-up figures.
- Audio/image execution is declared but not wired; `transport="declared"` +
  typed `ProviderError` is the honest runtime contract until those transports
  are implemented.

## Deliverables

- Catalog + registration: `providers/hf.py` (`HFModelSpec`, `HF_MODEL_CATALOG`,
  `build_hf_manifests`, `register_hf_resources`, `_validate_spec`)
- Schema extensions: `models.py`, `capability_registry.py` (audio, image resource classes; 33 new capability flags)
- Tests: `test_hf_provider.py`, `test_model_catalog.py` (run with
  `python test_hf_provider.py` / `python test_model_catalog.py`)
- Machine-readable registry data: `outputs/registry_data.json`
  (generated by `registry_export.py`)

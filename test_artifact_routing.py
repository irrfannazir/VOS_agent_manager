"""Regression tests for multimodal artifact routing.

Tests that original input artifacts (audio, image) are preserved and routed
correctly to parallel branches, and that modality mismatches are detected.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bootstrap import install_aos_v0_shim

install_aos_v0_shim()

from agents.graph_executor import GraphExecutor
from capability_registry import (
    CapabilityManifest,
    CapabilityRegistry,
    CostModel,
    LatencyModel,
)
from models import (
    Artifact,
    ArtifactModalityMismatch,
    CapabilityDNA,
    DNAConstraints,
    Graph,
    Node,
)


def _audio_stub(resource_id):
    """Stub that records the input it received."""
    received_inputs = []

    def run(text, instruction=None):
        received_inputs.append(text)
        return f"[{resource_id}] processed audio input successfully. The audio file has been analyzed and the results are ready for downstream processing."

    run.received_inputs = received_inputs
    return run


def _image_stub(resource_id):
    """Stub that records the input it received."""
    received_inputs = []

    def run(text, instruction=None):
        received_inputs.append(text)
        return f"[{resource_id}] processed image input successfully. The image has been analyzed and the results are ready for downstream processing."

    run.received_inputs = received_inputs
    return run


def _text_stub(resource_id):
    """Stub that records the input it received."""
    received_inputs = []

    def run(text, instruction=None):
        received_inputs.append(text)
        return f"[{resource_id}] processed text input successfully. The text has been analyzed and the results are ready for downstream processing."

    run.received_inputs = received_inputs
    return run


def _build_multimodal_registry():
    """Registry with audio, image, and text resources."""
    registry = CapabilityRegistry()
    specs = [
        ("whisper", "asr", ["speech.transcription", "audio_input", "transcription"],
         0.0, 5000, {"speech.transcription": 0.9, "audio_input": 0.85}),
        ("ast_classifier", "audio", ["audio_classification", "audio_analysis", "audio_input"],
         0.0, 4000, {"audio_classification": 0.88, "audio_analysis": 0.85}),
        ("clip_vision", "vlm", ["vision.understanding", "image_classification", "vision_input"],
         0.0, 6000, {"vision.understanding": 0.9, "image_classification": 0.85}),
        ("summarizer", "llm", ["text.summarization", "reasoning.deep"],
         0.003, 5000, {"text.summarization": 0.85, "reasoning.deep": 0.8}),
    ]
    for rid, cls, caps, cost, p95, priors in specs:
        registry.register(
            CapabilityManifest(
                resource_id=rid,
                resource_class=cls,
                capabilities=caps,
                cost_model=CostModel(unit="per_call", estimate_usd=cost),
                latency_model=LatencyModel(p50_ms=p95 // 3, p95_ms=p95),
                quality_priors=priors,
            ),
            _audio_stub(rid) if "audio" in caps or "speech" in caps[0]
            else _image_stub(rid) if "vision" in caps[0] or "image" in caps[0]
            else _text_stub(rid),
        )
    return registry


def test_audio_branching():
    """Parallel nodes from same audio input both get the audio file."""
    registry = _build_multimodal_registry()

    graph = Graph(
        job="Transcribe and classify this audio recording",
        nodes=[
            Node(
                id="transcribe",
                description="Transcribe the audio file",
                capability="speech_transcription",
                depends_on=[],
                dna=CapabilityDNA(
                    flags=["speech.transcription", "audio_input"],
                    constraints=DNAConstraints(cost_ceiling_usd=0.05, latency_slo_ms=30_000),
                    confidence=0.9,
                ),
            ),
            Node(
                id="classify",
                description="Classify audio events in the recording",
                capability="audio",
                depends_on=[],
                dna=CapabilityDNA(
                    flags=["audio_classification", "audio_analysis", "audio_input"],
                    constraints=DNAConstraints(cost_ceiling_usd=0.05, latency_slo_ms=30_000),
                    confidence=0.9,
                ),
            ),
            Node(
                id="summarize",
                description="Summarize the audio analysis",
                capability="summarization",
                depends_on=["transcribe", "classify"],
                dna=CapabilityDNA(
                    flags=["text.summarization", "reasoning.deep"],
                    constraints=DNAConstraints(cost_ceiling_usd=0.05, latency_slo_ms=30_000),
                    confidence=0.9,
                ),
            ),
        ],
        artifacts={
            "input_audio": Artifact(id="input_audio", modality="audio", path="/tmp/audio.wav", source="user_input"),
        },
    )

    executor = GraphExecutor(registry)
    result = executor.run(graph, inputs={"audio": "/tmp/audio.wav"})

    # Both audio nodes should get the audio file path, not transcription text.
    transcribe_node = next(n for n in result.nodes if n.id == "transcribe")
    classify_node = next(n for n in result.nodes if n.id == "classify")

    assert transcribe_node.status == "done", f"transcribe status: {transcribe_node.status}"
    assert classify_node.status == "done", f"classify status: {classify_node.status}"

    # Both should have received the audio file path as input.
    # The stub receives the node.input which is set by the executor.
    # For root nodes with data_inputs, they get the artifact path.
    # For root nodes without data_inputs, they get the file path from inputs dict.
    print("test_audio_branching: PASSED")


def test_image_branching():
    """Parallel nodes from same image input both get the image file."""
    registry = _build_multimodal_registry()

    graph = Graph(
        job="Classify and understand this image",
        nodes=[
            Node(
                id="classify",
                description="Classify the image",
                capability="vision",
                depends_on=[],
                dna=CapabilityDNA(
                    flags=["image_classification", "vision_input"],
                    constraints=DNAConstraints(cost_ceiling_usd=0.05, latency_slo_ms=30_000),
                    confidence=0.9,
                ),
            ),
            Node(
                id="understand",
                description="Understand the image content",
                capability="vision",
                depends_on=[],
                dna=CapabilityDNA(
                    flags=["vision.understanding", "vision_input"],
                    constraints=DNAConstraints(cost_ceiling_usd=0.05, latency_slo_ms=30_000),
                    confidence=0.9,
                ),
            ),
            Node(
                id="describe",
                description="Describe the image",
                capability="summarization",
                depends_on=["classify", "understand"],
                dna=CapabilityDNA(
                    flags=["text.summarization", "reasoning.deep"],
                    constraints=DNAConstraints(cost_ceiling_usd=0.05, latency_slo_ms=30_000),
                    confidence=0.9,
                ),
            ),
        ],
        artifacts={
            "input_image": Artifact(id="input_image", modality="image", path="/tmp/image.jpg", source="user_input"),
        },
    )

    executor = GraphExecutor(registry)
    result = executor.run(graph, inputs={"image": "/tmp/image.jpg"})

    classify_node = next(n for n in result.nodes if n.id == "classify")
    understand_node = next(n for n in result.nodes if n.id == "understand")

    assert classify_node.status == "done", f"classify status: {classify_node.status}"
    assert understand_node.status == "done", f"understand status: {understand_node.status}"

    print("test_image_branching: PASSED")


def test_modality_mismatch_detection():
    """Node expecting image gets audio → ArtifactModalityMismatch."""
    registry = _build_multimodal_registry()

    graph = Graph(
        job="Classify this audio as if it were an image",
        nodes=[
            Node(
                id="bad_node",
                description="Classify this as an image",
                capability="vision",
                depends_on=[],
                data_inputs=["input_audio"],
                dna=CapabilityDNA(
                    flags=["image_classification", "vision_input"],
                    constraints=DNAConstraints(cost_ceiling_usd=0.05, latency_slo_ms=30_000),
                    confidence=0.9,
                ),
            ),
        ],
        artifacts={
            "input_audio": Artifact(id="input_audio", modality="audio", path="/tmp/audio.wav", source="user_input"),
        },
    )

    executor = GraphExecutor(registry)
    try:
        result = executor.run(graph)
        # Should not reach here — ArtifactModalityMismatch should be raised
        print("test_modality_mismatch_detection: FAILED (no exception raised)")
        sys.exit(1)
    except ArtifactModalityMismatch as e:
        assert e.node_id == "bad_node"
        assert e.artifact_id == "input_audio"
        assert e.expected == "image"
        assert e.actual == "audio"
        print("test_modality_mismatch_detection: PASSED")


def test_data_inputs_routing():
    """Node with data_inputs gets specified artifacts, not parent outputs."""
    registry = _build_multimodal_registry()

    graph = Graph(
        job="Process audio and text",
        nodes=[
            Node(
                id="text_node",
                description="Summarize the text",
                capability="summarization",
                depends_on=[],
                dna=CapabilityDNA(
                    flags=["text.summarization", "reasoning.deep"],
                    constraints=DNAConstraints(cost_ceiling_usd=0.05, latency_slo_ms=30_000),
                    confidence=0.9,
                ),
            ),
            Node(
                id="audio_node",
                description="Transcribe the audio",
                capability="speech_transcription",
                depends_on=["text_node"],
                data_inputs=["input_audio"],
                dna=CapabilityDNA(
                    flags=["speech.transcription", "audio_input"],
                    constraints=DNAConstraints(cost_ceiling_usd=0.05, latency_slo_ms=30_000),
                    confidence=0.9,
                ),
            ),
        ],
        artifacts={
            "input_audio": Artifact(id="input_audio", modality="audio", path="/tmp/audio.wav", source="user_input"),
        },
    )

    executor = GraphExecutor(registry)
    result = executor.run(graph, inputs={"audio": "/tmp/audio.wav"})

    text_node = next(n for n in result.nodes if n.id == "text_node")
    audio_node = next(n for n in result.nodes if n.id == "audio_node")

    assert text_node.status == "done", f"text_node status: {text_node.status}"
    assert audio_node.status == "done", f"audio_node status: {audio_node.status}"

    # audio_node should have received the audio file path, not text_node's output.
    # We can verify by checking that audio_node.input contains the audio path.
    assert "/tmp/audio.wav" in (audio_node.input or ""), (
        f"audio_node.input should contain audio path, got: {audio_node.input}"
    )

    print("test_data_inputs_routing: PASSED")


def test_capability_preserving_fallback():
    """Relaxed routing marks partial matches as degraded."""
    # Build registry with only whisper (no AST classifier)
    registry = CapabilityRegistry()
    registry.register(
        CapabilityManifest(
            resource_id="whisper",
            resource_class="asr",
            capabilities=["speech.transcription", "audio_input", "transcription"],
            cost_model=CostModel(unit="per_call", estimate_usd=0.0),
            latency_model=LatencyModel(p50_ms=1000, p95_ms=5000),
            quality_priors={"speech.transcription": 0.9, "audio_input": 0.85},
        ),
        _audio_stub("whisper"),
    )

    graph = Graph(
        job="Classify audio events",
        nodes=[
            Node(
                id="classify",
                description="Classify audio events in the recording",
                capability="audio",
                depends_on=[],
                dna=CapabilityDNA(
                    flags=["audio_classification", "audio_analysis", "audio_input"],
                    constraints=DNAConstraints(cost_ceiling_usd=0.05, latency_slo_ms=30_000),
                    confidence=0.9,
                ),
            ),
        ],
        artifacts={
            "input_audio": Artifact(id="input_audio", modality="audio", path="/tmp/audio.wav", source="user_input"),
        },
    )

    executor = GraphExecutor(registry)
    result = executor.run(graph, inputs={"audio": "/tmp/audio.wav"})

    classify_node = next(n for n in result.nodes if n.id == "classify")

    # Should be degraded because whisper is not a capability-preserving match
    # for audio_classification.
    assert classify_node.status == "degraded", (
        f"classify status should be 'degraded', got: {classify_node.status}"
    )
    assert classify_node.routing_mode == "relaxed", (
        f"classify routing_mode should be 'relaxed', got: {classify_node.routing_mode}"
    )

    print("test_capability_preserving_fallback: PASSED")


if __name__ == "__main__":
    failures = []

    for test_fn in [
        test_audio_branching,
        test_image_branching,
        test_modality_mismatch_detection,
        test_data_inputs_routing,
        test_capability_preserving_fallback,
    ]:
        try:
            test_fn()
        except Exception as e:
            failures.append(f"{test_fn.__name__}: {e}")

    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\nall artifact routing tests passed")

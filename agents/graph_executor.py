import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from aos_v0.capability_registry import CapabilityRegistry
from aos_v0.failure_manager import FailureManager
from aos_v0.graph_utils import build_waves
from aos_v0.models import Artifact, ArtifactModalityMismatch, Graph
from aos_v0.agents.sub_agent import SubAgent

_print_lock = threading.Lock()

_AUDIO_FLAGS = frozenset({
    "speech.transcription", "speech_recognition",
    "automatic_speech_recognition", "transcription",
    "audio_input", "audio_understanding",
    "audio_analysis", "audio_to_text",
    "multilingual_speech", "speech_understanding",
})

_IMAGE_FLAGS = frozenset({
    "vision.understanding", "vision_input",
    "image_classification", "object_detection",
    "object_identification", "object_localization",
    "multi_object_detection", "image_understanding",
    "visual_question_answering", "visual_reasoning",
    "vision_language",
})


def _safe_print(msg: str) -> None:
    with _print_lock:
        print(msg)


def _infer_modality(node_flags: set, capability: str) -> Optional[str]:
    """Infer the file modality a node needs from its DNA flags and capability."""
    if node_flags & _AUDIO_FLAGS or capability in ("speech_transcription", "audio"):
        return "audio"
    if node_flags & _IMAGE_FLAGS or capability == "vision":
        return "image"
    return None


class GraphExecutor:
    """Runs the DAG wave by wave, handing each node to a SubAgent.

    The registry is held here rather than built per node so every sub-agent
    shares one manifest table -- the Learning Manager's prior updates would
    otherwise be lost between nodes. The FailureManager is shared for the same
    reason: one fault report per workflow, not one per node.
    """

    def __init__(
        self,
        registry: CapabilityRegistry,
        failure_manager: Optional[FailureManager] = None,
    ):
        self.registry = registry
        self.failure_manager = failure_manager or FailureManager(registry)

    def run(self, graph: Graph, inputs: Optional[dict[str, str]] = None) -> Graph:
        waves = build_waves(graph)
        by_id = {n.id: n for n in graph.nodes}
        inputs = inputs or {}

        # --- artifact lineage tracking ---
        # Register user-supplied input artifacts so parallel branches from the
        # same original both receive the correct file.
        artifacts: dict[str, Artifact] = dict(graph.artifacts)
        for modality, path in inputs.items():
            if modality in ("audio", "image") and path:
                aid = f"input_{modality}"
                if aid not in artifacts:
                    artifacts[aid] = Artifact(
                        id=aid, modality=modality, path=path, source="user_input",
                    )
        graph.artifacts = artifacts

        for wave_idx, wave in enumerate(waves):
            _safe_print(
                f"[graph-executor] wave {wave_idx} starting "
                f"({len(wave)} nodes, running concurrently)"
            )

            def _run_node(node):
                # --- determine node input ---
                if node.data_inputs:
                    # Explicit data_inputs: route specified artifacts, regardless
                    # of whether this is a root node or a dependent node.
                    parts = []
                    for aid in node.data_inputs:
                        art = artifacts.get(aid)
                        if art is None:
                            parts.append(f"[artifact '{aid}' not found]")
                            continue
                        self._validate_modality(node, art)
                        content = self._read_artifact(art)
                        parts.append(
                            f"From artifact '{aid}' ({art.modality}):\n{content}"
                        )
                    node.input = "\n\n".join(parts)
                elif node.depends_on:
                    # Dependent node without data_inputs: use parent outputs.
                    parts = []
                    for parent_id in node.depends_on:
                        parent = by_id[parent_id]
                        parts.append(
                            f"From node '{parent.id}' ({parent.description}):\n"
                            f"{parent.output or '(no output produced)'}"
                        )
                    node.input = "\n\n".join(parts)
                else:
                    # Root node without data_inputs: route typed input artifacts
                    # based on DNA flags.
                    node_flags = set(node.dna.flags) if node.dna else set()
                    modality = _infer_modality(node_flags, node.capability)
                    if modality and modality in inputs:
                        node.input = inputs[modality]
                    else:
                        node.input = graph.job

                # Inject vision context for non-vision downstream nodes.
                vision_arts = [
                    a for a in artifacts.values()
                    if a.modality == "image" and a.source != "user_input"
                ]
                if node.capability != "vision" and vision_arts:
                    vision_ctx = "\n\n".join(
                        f"[IMAGE IDENTIFICATION]: {self._read_artifact(a)}"
                        for a in vision_arts
                    )
                    node.input = (
                        f"IMPORTANT CONTEXT — an image was analyzed and the following "
                        f"was identified:\n{vision_ctx}\n\n---\n\n{node.input}"
                    )

                # Inject audio context for non-audio downstream nodes.
                audio_arts = [
                    a for a in artifacts.values()
                    if a.modality == "audio" and a.source != "user_input"
                ]
                if node.capability not in ("speech_transcription", "audio") and audio_arts:
                    audio_ctx = "\n\n".join(
                        f"[AUDIO TRANSCRIPTION]: {self._read_artifact(a)}"
                        for a in audio_arts
                    )
                    node.input = (
                        f"IMPORTANT CONTEXT — an audio file was transcribed and the following "
                        f"was identified:\n{audio_ctx}\n\n---\n\n{node.input}"
                    )

                agent = SubAgent(
                    name=f"sub-agent-{node.id}",
                    capability=node.capability,
                    registry=self.registry,
                    failure_manager=self.failure_manager,
                )
                agent.perform(node)

                # Register node output as a new artifact for downstream nodes.
                if node.output and node.status != "degraded":
                    node_flags = set(node.dna.flags) if node.dna else set()
                    modality = _infer_modality(node_flags, node.capability)
                    out_modality = modality or "text"
                    out_aid = f"output_{node.id}"
                    artifacts[out_aid] = Artifact(
                        id=out_aid,
                        modality=out_modality,
                        source=f"node:{node.id}",
                    )

                _safe_print(
                    f"[graph-executor] node '{node.id}' {node.status} "
                    f"({len(node.output or '')} chars)"
                )
                return node

            with ThreadPoolExecutor(max_workers=len(wave)) as pool:
                futures = {pool.submit(_run_node, node): node for node in wave}
                for future in as_completed(futures):
                    future.result()

            _safe_print(f"[graph-executor] wave {wave_idx} complete")

        return graph

    # -- modality validation -------------------------------------------------

    @staticmethod
    def _validate_modality(node, artifact: Artifact) -> None:
        """Raise ArtifactModalityMismatch if the artifact doesn't match the node's input requirement."""
        node_flags = set(node.dna.flags) if node.dna else set()
        expected = _infer_modality(node_flags, node.capability)
        if expected and artifact.modality != expected:
            raise ArtifactModalityMismatch(
                node.id, artifact.id, expected, artifact.modality,
            )

    @staticmethod
    def _read_artifact(artifact: Artifact) -> str:
        """Read artifact content. For file artifacts, return the path so providers can load it."""
        if artifact.path:
            return artifact.path
        return f"[artifact {artifact.id} ({artifact.modality})]"

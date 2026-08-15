"""End-to-end executor test with stub resources -- no network, no LLM.

Proves the wave scheduler, DNA routing and the exact-match degradation path all
work together. The old version of this file pointed at a v0.0.2 checkout and
constructed SubAgent without a registry; both are fixed here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bootstrap import install_aos_v0_shim

install_aos_v0_shim()

from agents.graph_executor import GraphExecutor
from agents.integrator_agent import IntegratorAgent
from capability_registry import (
    CapabilityManifest,
    CapabilityRegistry,
    CostModel,
    LatencyModel,
)
from models import CapabilityDNA, DNAConstraints, Graph, Node


def _stub(resource_id):
    def run(text, instruction=None):
        return f"[{resource_id}] {instruction}"

    return run


def build_stub_registry() -> CapabilityRegistry:
    registry = CapabilityRegistry()
    specs = [
        ("web_search", "tool", ["web.search", "tool.calling"], 0.004, 9000,
         {"web.search": 0.85, "tool.calling": 0.8}),
        ("summarization", "llm",
         ["text.summarization", "reasoning.deep", "reasoning.shallow"], 0.003, 5000,
         {"text.summarization": 0.85, "reasoning.deep": 0.8, "reasoning.shallow": 0.75}),
        ("quick_summarization", "llm", ["text.summarization", "text.classification"],
         0.001, 2200, {"text.summarization": 0.65, "text.classification": 0.8}),
        ("vision", "vlm", ["vision.understanding"], 0.0, 15000,
         {"vision.understanding": 0.88}),
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
            _stub(rid),
        )
    return registry


_LOOSE = DNAConstraints(cost_ceiling_usd=0.05, latency_slo_ms=30_000)
_TIGHT = DNAConstraints(cost_ceiling_usd=0.002, latency_slo_ms=30_000)

graph = Graph(
    job="Compare solar energy and wind energy as renewable sources.",
    nodes=[
        Node(
            id="a",
            description="Search for information about solar energy",
            capability="web_search",
            depends_on=[],
            dna=CapabilityDNA(flags=["web.search"], constraints=_LOOSE, confidence=0.9),
        ),
        Node(
            id="b",
            description="Search for information about wind energy",
            capability="web_search",
            depends_on=[],
            dna=CapabilityDNA(flags=["web.search"], constraints=_LOOSE, confidence=0.9),
        ),
        # Tight budget -> should degrade to the cheap summarizer.
        Node(
            id="c",
            description="Summarize findings about solar energy",
            capability="summarization",
            depends_on=["a"],
            dna=CapabilityDNA(
                flags=["text.summarization"], constraints=_TIGHT, confidence=0.9
            ),
        ),
        # No DNA at all -> must fall back to exact-match routing.
        Node(
            id="d",
            description="Summarize findings about wind energy",
            capability="summarization",
            depends_on=["b"],
        ),
        # reasoning.deep is only on the full summarizer.
        Node(
            id="e",
            description="Compare solar and wind based on the summaries",
            capability="summarization",
            depends_on=["c", "d"],
            dna=CapabilityDNA(
                flags=["reasoning.deep", "text.summarization"],
                constraints=_LOOSE,
                confidence=0.9,
            ),
        ),
    ],
)

print("=" * 60)
print("Running 5-node graph against stub resources")
print("=" * 60)

result = GraphExecutor(build_stub_registry()).run(graph)
final = IntegratorAgent().integrate(result)

print("\n" + "=" * 60)
print("ROUTING RESULT")
print("=" * 60)
for n in result.nodes:
    print(
        f"  node '{n.id}': status={n.status} mode={n.routing_mode} "
        f"resource={n.bound_resource} output_len={len(n.output or '')}"
    )

expected = {
    "a": ("dna", "web_search"),
    "b": ("dna", "web_search"),
    "c": ("dna", "quick_summarization"),  # tight budget filters the full summarizer
    "d": ("exact", "summarization"),      # no DNA -> fallback path
    "e": ("dna", "summarization"),        # only resource with reasoning.deep
}

failures = []
for node in result.nodes:
    want_mode, want_resource = expected[node.id]
    if node.status != "done":
        failures.append(f"node '{node.id}' status={node.status}")
    if (node.routing_mode, node.bound_resource) != (want_mode, want_resource):
        failures.append(
            f"node '{node.id}' routed to ({node.routing_mode}, {node.bound_resource}), "
            f"expected ({want_mode}, {want_resource})"
        )

if not final:
    failures.append("integrator produced empty output")

print("\n" + "=" * 60)
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("all checks passed")

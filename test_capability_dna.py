"""Offline tests for Capability DNA, the feasibility filter and the Pareto scorer.

No network, no LLM: every resource is a stub, so this suite is the determinism
control for the routing layer. Run with `python test_capability_dna.py`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bootstrap import install_aos_v0_shim

install_aos_v0_shim()

from capability_registry import (
    Availability,
    CapabilityManifest,
    CapabilityRegistry,
    CostModel,
    InfeasibleDNAError,
    LatencyModel,
)
from constraint_policy import ConstraintPolicy
from dna_extractor import DNAExtractor
from models import CapabilityDNA, DNAConstraints, DNAOrdinals, Graph, Node

_failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label} {detail}")
        _failures.append(label)


def _stub(name):
    def run(text, instruction=None):
        return f"{name}:{instruction}"

    return run


def build_test_registry() -> CapabilityRegistry:
    """Two summarizers that differ only in quality/cost/latency, plus a vision
    resource and one that is down. Enough to exercise every filter branch."""
    registry = CapabilityRegistry()

    registry.register(
        CapabilityManifest(
            resource_id="strong_llm",
            resource_class="llm",
            capabilities=["text.summarization", "reasoning.deep", "reasoning.shallow"],
            cost_model=CostModel(unit="per_call", estimate_usd=0.003),
            latency_model=LatencyModel(p50_ms=1500, p95_ms=5000),
            quality_priors={
                "text.summarization": 0.85,
                "reasoning.deep": 0.80,
                "reasoning.shallow": 0.75,
            },
            risk_class="low",
        ),
        _stub("strong_llm"),
    )
    registry.register(
        CapabilityManifest(
            resource_id="cheap_llm",
            resource_class="llm",
            capabilities=["text.summarization", "reasoning.shallow"],
            cost_model=CostModel(unit="per_call", estimate_usd=0.0005),
            latency_model=LatencyModel(p50_ms=400, p95_ms=1200),
            quality_priors={"text.summarization": 0.60, "reasoning.shallow": 0.65},
            risk_class="low",
        ),
        _stub("cheap_llm"),
    )
    registry.register(
        CapabilityManifest(
            resource_id="risky_search",
            resource_class="tool",
            capabilities=["web.search", "tool.calling"],
            cost_model=CostModel(unit="per_call", estimate_usd=0.004),
            latency_model=LatencyModel(p50_ms=3000, p95_ms=9000),
            quality_priors={"web.search": 0.9, "tool.calling": 0.85},
            risk_class="high",
        ),
        _stub("risky_search"),
    )
    registry.register(
        CapabilityManifest(
            resource_id="dead_vlm",
            resource_class="vlm",
            capabilities=["vision.understanding"],
            quality_priors={"vision.understanding": 0.99},
            availability=Availability(status="down"),
        ),
        _stub("dead_vlm"),
    )
    return registry


def test_dna_validation() -> None:
    print("\n[dna] schema validation")

    try:
        CapabilityDNA(flags=["not.a.real.flag"])
        check("rejects out-of-vocabulary flag", False, "no error raised")
    except Exception:
        check("rejects out-of-vocabulary flag", True)

    dna = CapabilityDNA(flags=["reasoning.deep", "reasoning.deep"])
    check("de-duplicates flags", dna.flags == ["reasoning.deep"], f"got {dna.flags}")

    try:
        DNAOrdinals(reasoning_depth=7)
        check("clamps ordinals to 0-4", False, "no error raised")
    except Exception:
        check("clamps ordinals to 0-4", True)

    # demand = (4 + 4 + 4) / 12 = 1.0 -> effective floor overrides the declared 0.0
    hard = CapabilityDNA(
        ordinals=DNAOrdinals(reasoning_depth=4, planning_horizon=4, tool_complexity=4),
        constraints=DNAConstraints(min_quality=0.0),
    )
    check(
        "ordinals raise effective min_quality",
        hard.effective_min_quality() == 1.0,
        f"got {hard.effective_min_quality()}",
    )

    # parallelizability is a scheduling hint and must NOT inflate demand
    par = CapabilityDNA(ordinals=DNAOrdinals(parallelizability=4, memory_dependence=4))
    check(
        "parallelizability/memory excluded from demand",
        par.ordinals.demand() == 0.0,
        f"got {par.ordinals.demand()}",
    )


def test_feasibility_filter() -> None:
    print("\n[m5 stage 1] feasibility filter")
    registry = build_test_registry()

    dna = CapabilityDNA(flags=["reasoning.deep"])
    survivors, rejections = registry.feasible(dna)
    check(
        "set containment excludes resources missing a flag",
        [m.resource_id for m in survivors] == ["strong_llm"],
        f"got {[m.resource_id for m in survivors]}",
    )
    check(
        "rejection names the missing capability",
        "missing capability" in rejections["cheap_llm"],
        f"got {rejections['cheap_llm']!r}",
    )

    dna = CapabilityDNA(flags=["vision.understanding"])
    _, rejections = registry.feasible(dna)
    check(
        "down resource rejected on availability",
        "unavailable" in rejections["dead_vlm"],
        f"got {rejections['dead_vlm']!r}",
    )

    dna = CapabilityDNA(
        flags=["web.search"],
        constraints=DNAConstraints(risk_tolerance="low"),
    )
    survivors, rejections = registry.feasible(dna)
    check(
        "risk_class above tolerance is filtered",
        not survivors and "risk_class" in rejections["risky_search"],
        f"got {rejections.get('risky_search')!r}",
    )

    dna = CapabilityDNA(
        flags=["text.summarization"],
        constraints=DNAConstraints(cost_ceiling_usd=0.001),
    )
    survivors, rejections = registry.feasible(dna)
    check(
        "cost ceiling filters the expensive resource",
        [m.resource_id for m in survivors] == ["cheap_llm"],
        f"got {[m.resource_id for m in survivors]}",
    )
    check(
        "cost rejection reports the overrun",
        "over ceiling" in rejections["strong_llm"],
        f"got {rejections['strong_llm']!r}",
    )

    dna = CapabilityDNA(
        flags=["text.summarization"],
        constraints=DNAConstraints(latency_slo_ms=2000),
    )
    survivors, _ = registry.feasible(dna)
    check(
        "latency SLO filters the slow resource",
        [m.resource_id for m in survivors] == ["cheap_llm"],
        f"got {[m.resource_id for m in survivors]}",
    )

    dna = CapabilityDNA(
        flags=["text.summarization"],
        constraints=DNAConstraints(min_quality=0.8),
    )
    survivors, _ = registry.feasible(dna)
    check(
        "min_quality filters the weak resource",
        [m.resource_id for m in survivors] == ["strong_llm"],
        f"got {[m.resource_id for m in survivors]}",
    )


def test_pareto_scorer() -> None:
    print("\n[m5 stage 2] Pareto scorer")
    registry = build_test_registry()

    # Loose budget: quality dominates, so the strong resource should win.
    loose = CapabilityDNA(
        flags=["text.summarization"],
        constraints=DNAConstraints(cost_ceiling_usd=0.05, latency_slo_ms=30_000),
    )
    decision = registry.select(loose)
    check(
        "loose budget picks the high-quality resource",
        decision.resource_id == "strong_llm",
        f"got {decision.resource_id}",
    )
    check(
        "runner-up recorded with a margin",
        decision.runner_up == "cheap_llm" and decision.runner_up_margin > 0,
        f"got {decision.runner_up} / {decision.runner_up_margin}",
    )

    # Squeeze the SLO until the latency term outweighs the quality gap. The
    # strong resource is still feasible (p95 5000 <= 5200) but now expensive
    # in score terms -- this is the Pareto tradeoff, not a filter rejection.
    tight = CapabilityDNA(
        flags=["text.summarization"],
        constraints=DNAConstraints(cost_ceiling_usd=0.0031, latency_slo_ms=5200),
    )
    registry.lambda_cost = 0.3
    registry.mu_latency = 1.0
    decision = registry.select(tight)
    check(
        "tight SLO flips the choice to the cheap resource",
        decision.resource_id == "cheap_llm",
        f"got {decision.resource_id} (scores: "
        f"{[(c.resource_id, round(c.score, 3)) for c in decision.candidates]})",
    )
    check(
        "both resources survived the filter (tradeoff, not rejection)",
        len(decision.candidates) == 2,
        f"got {len(decision.candidates)}",
    )

    # Determinism: identical manifests must bind by resource_id tie-break.
    registry.lambda_cost, registry.mu_latency = 0.0, 0.0
    twin = CapabilityDNA(flags=["reasoning.shallow"])
    first = registry.select(twin).resource_id
    check(
        "repeated selection is deterministic",
        all(registry.select(twin).resource_id == first for _ in range(5)),
    )


def test_infeasible_and_admission() -> None:
    print("\n[admission] infeasible DNA")
    registry = build_test_registry()

    dna = CapabilityDNA(flags=["code.generation"])
    try:
        registry.select(dna)
        check("select raises on infeasible DNA", False, "no error raised")
    except InfeasibleDNAError as exc:
        check("select raises on infeasible DNA", True)
        check(
            "error names the offending flag",
            "code.generation" in str(exc),
            f"got {exc}",
        )

    check(
        "unsatisfiable_flags reports the gap",
        registry.unsatisfiable_flags(dna) == ["code.generation"],
        f"got {registry.unsatisfiable_flags(dna)}",
    )
    check(
        "satisfiable DNA reports no gap",
        registry.unsatisfiable_flags(CapabilityDNA(flags=["web.search"])) == [],
    )

    empty = CapabilityRegistry()
    try:
        empty.select(CapabilityDNA(flags=["web.search"]))
        check("empty registry raises", False, "no error raised")
    except InfeasibleDNAError:
        check("empty registry raises", True)


def test_constraint_policy() -> None:
    """The regression that motivated the module: an LLM-invented 500 ms SLO
    rejected every resource in the pool and DNA routing never fired."""
    print("\n[policy] kernel-derived constraints")
    registry = build_test_registry()
    policy = ConstraintPolicy(registry, job_budget_usd=0.50)

    dna = CapabilityDNA(flags=["text.summarization"])
    con = policy.derive(dna, node_count=10)
    check(
        "cost ceiling is the per-node share of the job budget",
        abs(con.cost_ceiling_usd - 0.05) < 1e-6,
        f"got {con.cost_ceiling_usd}",
    )
    check(
        "latency SLO clears the slowest capable resource",
        con.latency_slo_ms >= 5000,
        f"got {con.latency_slo_ms} (pool p95 5000)",
    )
    check(
        "derived constraints leave the DNA feasible",
        registry_select_ok(registry, CapabilityDNA(flags=dna.flags, constraints=con)),
    )

    # A ceiling below every resource's price would be a bug, not a budget.
    starved = ConstraintPolicy(registry, job_budget_usd=0.0001).derive(
        CapabilityDNA(flags=["text.summarization"]), node_count=100
    )
    check(
        "ceiling floors at the cheapest capable resource",
        starved.cost_ceiling_usd >= 0.0005,
        f"got {starved.cost_ceiling_usd}",
    )

    # web.search resources are risk_class=medium; a low tolerance would filter
    # out the only resource that can do the job.
    web = policy.derive(CapabilityDNA(flags=["web.search"]), node_count=4)
    check(
        "external-data flags get a permissive risk tolerance",
        web.risk_tolerance == "high",
        f"got {web.risk_tolerance}",
    )
    check(
        "internal-only flags stay risk-averse",
        policy.derive(
            CapabilityDNA(flags=["text.summarization"]), node_count=4
        ).risk_tolerance
        == "low",
    )

    # Harder nodes must demand better resources.
    easy = policy.derive(CapabilityDNA(ordinals=DNAOrdinals()), 4).min_quality
    hard = policy.derive(
        CapabilityDNA(
            ordinals=DNAOrdinals(
                reasoning_depth=4, planning_horizon=4, tool_complexity=4
            )
        ),
        4,
    ).min_quality
    check("ordinal demand raises min_quality", hard > easy, f"{easy} -> {hard}")

    # Every node in a realistic plan must stay routable end to end.
    graph = Graph(
        job="j",
        nodes=[
            Node(id="a", description="search", capability="web_search",
                 dna=CapabilityDNA(flags=["web.search"])),
            Node(id="b", description="summarize", capability="summarization",
                 depends_on=["a"], dna=CapabilityDNA(flags=["text.summarization"])),
            Node(id="c", description="answer", capability="synthesis",
                 depends_on=["b"], dna=CapabilityDNA(flags=["reasoning.deep"])),
        ],
    )
    ConstraintPolicy(registry, job_budget_usd=0.50).apply(graph)
    routable = all(
        registry_select_ok(registry, n.dna) for n in graph.nodes if n.dna
    )
    check("whole plan stays routable after policy applied", routable)


def test_heuristic_extractor() -> None:
    print("\n[m4] heuristic fallback (no network)")

    cases = [
        ("Describe the objects in this image", "vision", "vision.understanding"),
        ("Search for recent solar panel efficiency data", "web_search", "web.search"),
        ("Summarize the findings", "summarization", "text.summarization"),
        ("Compare solar and wind on cost", "summarization", "reasoning.deep"),
    ]
    for description, capability, expected in cases:
        node = Node(id="x", description=description, capability=capability)
        dna = DNAExtractor._heuristic_dna(node)
        check(
            f"heuristic maps {description[:28]!r} -> {expected}",
            expected in dna.flags,
            f"got {dna.flags}",
        )

    node = Node(id="x", description="Do the thing", capability="summarization")
    dna = DNAExtractor._heuristic_dna(node)
    check(
        "unmatched description falls back to the coarse capability",
        dna.flags == ["text.summarization"],
        f"got {dna.flags}",
    )
    check("heuristic DNA is marked low-confidence", dna.confidence <= 0.3)
    check(
        "heuristic constraints stay permissive (never self-blocks)",
        registry_accepts(dna),
    )

    # Out-of-vocabulary flags from a model must be dropped, not fatal. A
    # hallucinated `constraints` key must be ignored outright -- that field is
    # the kernel's, and a 500 ms SLO smuggled in here would break routing.
    dna = DNAExtractor._build_dna(
        {
            "flags": ["text.summarization", "hallucinated.flag"],
            "ordinals": {"reasoning_depth": 2, "invented_axis": 9},
            "constraints": {"cost_ceiling_usd": 0.05, "latency_slo_ms": 500},
            "confidence": 0.9,
        },
        extracted_by="test",
    )
    check(
        "invented flags dropped rather than raising",
        dna.flags == ["text.summarization"],
        f"got {dna.flags}",
    )
    check("partial ordinals default the rest", dna.ordinals.planning_horizon == 0)
    check("invented ordinal axes ignored", dna.ordinals.reasoning_depth == 2)
    check(
        "model-supplied constraints are ignored (kernel owns that segment)",
        dna.constraints.latency_slo_ms != 500,
        f"got {dna.constraints.latency_slo_ms}",
    )


def registry_accepts(dna: CapabilityDNA) -> bool:
    registry = build_test_registry()
    try:
        registry.select(dna)
        return True
    except InfeasibleDNAError:
        return False


def test_default_registry_routing() -> None:
    print("\n[integration] default registry, no LLM calls")
    from resource_registration import build_default_registry

    registry = build_default_registry()

    # A compare node needs reasoning.deep, which only the full summarizer has.
    deep = CapabilityDNA(
        flags=["reasoning.deep", "text.summarization"],
        constraints=DNAConstraints(cost_ceiling_usd=0.05, latency_slo_ms=30_000),
    )
    check(
        "reasoning.deep node binds to the full summarizer",
        registry.select(deep).resource_id == "summarization",
        f"got {registry.select(deep).resource_id}",
    )

    # Tighten the budget below the full summarizer's cost -> cheap one wins.
    cheap = CapabilityDNA(
        flags=["text.summarization"],
        constraints=DNAConstraints(cost_ceiling_usd=0.002, latency_slo_ms=30_000),
    )
    check(
        "tight budget binds to quick_summarization",
        registry.select(cheap).resource_id == "quick_summarization",
        f"got {registry.select(cheap).resource_id}",
    )

    vis = CapabilityDNA(
        flags=["vision.understanding"],
        constraints=DNAConstraints(latency_slo_ms=30_000),
    )
    check(
        "vision flag binds to the vision resource",
        registry.select(vis).resource_id == "vision",
        f"got {registry.select(vis).resource_id}",
    )

    web = CapabilityDNA(
        flags=["web.search"],
        constraints=DNAConstraints(latency_slo_ms=30_000, risk_tolerance="medium"),
    )
    check(
        "web.search flag binds to the search tool",
        registry.select(web).resource_id == "web_search",
        f"got {registry.select(web).resource_id}",
    )
    check(
        "web_search excluded under low risk tolerance",
        not registry_select_ok(
            registry,
            CapabilityDNA(
                flags=["web.search"],
                constraints=DNAConstraints(
                    latency_slo_ms=30_000, risk_tolerance="low"
                ),
            ),
        ),
    )

    # The terminal node must reach the synthesis resource, not the summarizer.
    answer = CapabilityDNA(
        flags=["answer.synthesis", "reasoning.deep"],
        constraints=DNAConstraints(cost_ceiling_usd=0.05, latency_slo_ms=30_000),
    )
    check(
        "answer.synthesis binds to the synthesis resource",
        registry.select(answer).resource_id == "synthesis",
        f"got {registry.select(answer).resource_id}",
    )
    check(
        "only the synthesis resource provides answer.synthesis",
        registry.find(["answer.synthesis"]) == ["synthesis"],
        f"got {registry.find(['answer.synthesis'])}",
    )


def registry_select_ok(registry, dna) -> bool:
    try:
        registry.select(dna)
        return True
    except InfeasibleDNAError:
        return False


if __name__ == "__main__":
    test_dna_validation()
    test_feasibility_filter()
    test_pareto_scorer()
    test_infeasible_and_admission()
    test_constraint_policy()
    test_heuristic_extractor()
    test_default_registry_routing()

    print("\n" + "=" * 60)
    if _failures:
        print(f"{len(_failures)} FAILURE(S):")
        for name in _failures:
            print(f"  - {name}")
        sys.exit(1)
    print("all checks passed")

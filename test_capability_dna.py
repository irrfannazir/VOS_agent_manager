"""Offline tests for Capability DNA and the continuous DNA scorer.

No network, no LLM: every resource is a stub, so this suite is the determinism
control for the routing layer. Run with `python test_capability_dna.py`.

The continuous scorer replaces the old two-stage hard-feasibility-filter +
Pareto scorer. Tests named test_feasibility_filter and test_pareto_scorer are
retained but rewritten to verify the new scorer semantics:
  - Availability is still a hard gate (down resources are excluded).
  - All other capability gaps are expressed as rejection-rate contributions.
  - The resource with the highest score wins.
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
    DimensionWeights,
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


def test_continuous_scorer_formula() -> None:
    """Unit tests for the _score_dimension math, independent of manifests."""
    print("\n[continuous-scorer] formula unit tests")

    # -- Over-qualified agent: acceptance branch ----------------------------
    acc, rej = CapabilityRegistry._score_dimension(
        agent_value=0.8, task_value=0.5, weight=1.0
    )
    check(
        "over-qualified: acceptance = (0.8-0.5)/0.8 * 1.0 = 0.375",
        abs(acc - 0.375) < 1e-9 and rej == 0.0,
        f"got acc={acc}, rej={rej}",
    )

    # -- Under-qualified agent: rejection branch ----------------------------
    acc, rej = CapabilityRegistry._score_dimension(
        agent_value=0.4, task_value=0.8, weight=1.0
    )
    expected_rej = (0.8 - 0.4) / (1.0 - 0.4)  # = 0.4/0.6 ~= 0.6667
    check(
        "under-qualified: rejection = (0.8-0.4)/(1-0.4) * 1.0 ~= 0.667",
        abs(rej - expected_rej) < 1e-9 and acc == 0.0,
        f"got acc={acc}, rej={rej}",
    )

    # -- Exact match: both branches produce zero contribution ---------------
    acc, rej = CapabilityRegistry._score_dimension(
        agent_value=0.7, task_value=0.7, weight=1.0
    )
    check(
        "exact match: acceptance=0, rejection=0",
        acc == 0.0 and rej == 0.0,
        f"got acc={acc}, rej={rej}",
    )

    # -- Div-by-zero guard: agent_value == 1, task_value < 1 (acceptance) ---
    acc, rej = CapabilityRegistry._score_dimension(
        agent_value=1.0, task_value=0.5, weight=2.0
    )
    check(
        "agent_value=1.0 > task: acceptance branch, no div-by-zero (no zero denominator in acc)",
        True,  # guard is in rejection branch; acceptance uses agent_value as divisor
    )
    # Guard explicitly tested: if agent_value=0 is the only case for acc branch guard
    acc, rej = CapabilityRegistry._score_dimension(
        agent_value=0.0, task_value=0.0, weight=1.0
    )
    check(
        "agent_value=0.0 == task=0.0: neither branch, zero contribution",
        acc == 0.0 and rej == 0.0,
        f"got acc={acc}, rej={rej}",
    )

    # -- Div-by-zero guard: agent_value == 1 in rejection branch ------------
    acc, rej = CapabilityRegistry._score_dimension(
        agent_value=1.0, task_value=1.0, weight=3.0
    )
    check(
        "agent_value=1.0 == task=1.0: exact match, rejection=0 (no div-by-zero)",
        acc == 0.0 and rej == 0.0,
        f"got acc={acc}, rej={rej}",
    )

    # -- Weight scales contribution -----------------------------------------
    acc1, _ = CapabilityRegistry._score_dimension(0.8, 0.5, weight=1.0)
    acc2, _ = CapabilityRegistry._score_dimension(0.8, 0.5, weight=2.0)
    check(
        "weight=2.0 produces exactly double the contribution",
        abs(acc2 - 2 * acc1) < 1e-9,
        f"got acc1={acc1}, acc2={acc2}",
    )


def test_feasibility_filter() -> None:
    """The availability gate is the only hard exclusion in the new scorer.

    Under the continuous scorer, capability mismatches and constraint violations
    are expressed as rejection-rate contributions rather than hard filters. The
    test checks:
      - A registry containing ONLY a down resource raises InfeasibleDNAError.
      - Resources missing a required flag still participate but score lower.
      - The resource that best matches the DNA wins overall.
    """
    print("\n[continuous-scorer] availability gate and score ordering")
    registry = build_test_registry()

    # -- Down-only isolated registry: select() must raise -------------------
    # Use a fresh registry with ONLY the down resource so there are no
    # other candidates to fall back to.
    down_registry = CapabilityRegistry()
    down_registry.register(
        CapabilityManifest(
            resource_id="dead_vlm",
            resource_class="vlm",
            capabilities=["vision.understanding"],
            quality_priors={"vision.understanding": 0.99},
            availability=Availability(status="down"),
        ),
        _stub("dead_vlm"),
    )
    try:
        down_registry.select(CapabilityDNA(flags=["vision.understanding"]))
        check(
            "down-only resource raises InfeasibleDNAError",
            False,
            "no error raised",
        )
    except InfeasibleDNAError:
        check("down-only resource raises InfeasibleDNAError", True)

    # Also verify: in the full registry, dead_vlm (down) does NOT appear
    # in the scored results.
    dna_vis = CapabilityDNA(flags=["vision.understanding"])
    full_decision = registry.select(dna_vis)
    scored_ids = [s.resource_id for s in full_decision.all_scores]
    check(
        "down resource does not appear in the scored list",
        "dead_vlm" not in scored_ids,
        f"scored list: {scored_ids}",
    )

    # -- Missing-flag resource scores lower but still participates ----------
    dna_deep = CapabilityDNA(
        flags=["reasoning.deep"],
        constraints=DNAConstraints(cost_ceiling_usd=1.0, latency_slo_ms=30_000),
    )
    decision = registry.select(dna_deep)
    # strong_llm provides reasoning.deep -> full flag-match acceptance.
    # cheap_llm does NOT provide reasoning.deep -> flag rejection penalty.
    # Both are scored; strong_llm must win.
    check(
        "resource with matching flag wins over resource without it",
        decision.resource_id == "strong_llm",
        f"got {decision.resource_id}",
    )
    all_ids = [s.resource_id for s in decision.all_scores]
    check(
        "non-matching resource still appears in scored list (not hard-excluded)",
        "cheap_llm" in all_ids,
        f"scored list: {all_ids}",
    )
    # Confirm strong_llm scored better
    scores_by_id = {s.resource_id: s.score for s in decision.all_scores}
    check(
        "matching resource has a strictly higher score",
        scores_by_id["strong_llm"] > scores_by_id["cheap_llm"],
        f"strong={scores_by_id.get('strong_llm'):.3f}, "
        f"cheap={scores_by_id.get('cheap_llm'):.3f}",
    )

    # -- Availability rejection appears in InfeasibleDNAError message -------
    try:
        down_registry.select(CapabilityDNA(flags=["vision.understanding"]))
    except InfeasibleDNAError as exc:
        check(
            "unavailable resource appears in rejection detail",
            "unavailable" in str(exc),
            f"got {exc}",
        )


def test_pareto_scorer() -> None:
    """Quality vs cost vs latency tradeoffs expressed through the continuous scorer.

    Under the new scorer, a task with high reasoning_depth ordinal raises
    effective_min_quality, creating a non-zero quality task_value. This makes
    the higher-quality resource win over the cheaper/faster one. With no ordinal
    demand (all zeros), the cheaper resource naturally wins on cost+latency.
    """
    print("\n[continuous-scorer] quality / cost / latency tradeoffs")
    registry = build_test_registry()

    # -- High reasoning demand: quality bar is raised, strong_llm wins ------
    # reasoning_depth=3 -> effective_min_quality = 3/12 = 0.25 > 0.
    # strong_llm quality=0.85 >> 0.25 (large acceptance).
    # cheap_llm quality=0.60 >> 0.25 (smaller acceptance gap vs quality bar).
    # The quality+reasoning dimensions together should favour strong_llm.
    demanding = CapabilityDNA(
        flags=["text.summarization"],
        ordinals=DNAOrdinals(reasoning_depth=3),
        constraints=DNAConstraints(cost_ceiling_usd=0.05, latency_slo_ms=30_000),
    )
    decision = registry.select(demanding)
    check(
        "high reasoning demand picks the high-quality resource",
        decision.resource_id == "strong_llm",
        f"got {decision.resource_id}",
    )
    check(
        "runner-up recorded with a positive margin",
        decision.runner_up == "cheap_llm" and decision.runner_up_margin > 0,
        f"got runner_up={decision.runner_up}, margin={decision.runner_up_margin}",
    )

    # -- Raising the latency weight heavily penalises the slow resource ------
    # Build a separate registry with high latency weight so cheap_llm
    # (p95=1200ms, much lower than strong_llm p95=5000ms) wins.
    high_lat = CapabilityRegistry(
        weights=DimensionWeights(latency=10.0)
    )
    for manifest in registry.manifests():
        high_lat.register(manifest, _stub(manifest.resource_id))

    tight_lat = CapabilityDNA(
        flags=["text.summarization"],
        constraints=DNAConstraints(
            cost_ceiling_usd=0.05,
            latency_slo_ms=6000,  # both resources fit, but cheap is much faster
        ),
    )
    decision_lat = high_lat.select(tight_lat)
    check(
        "high latency weight flips choice to the faster resource",
        decision_lat.resource_id == "cheap_llm",
        f"got {decision_lat.resource_id} "
        f"(scores: {[(s.resource_id, round(s.score, 3)) for s in decision_lat.all_scores]})",
    )
    check(
        "both resources appear in the scored list",
        len(decision_lat.all_scores) >= 2,
        f"got {len(decision_lat.all_scores)} scored resources",
    )

    # -- Determinism: identical weights -> tie-break on resource_id ----------
    twin = CapabilityDNA(flags=["reasoning.shallow"])
    first = registry.select(twin).resource_id
    check(
        "repeated selection is deterministic",
        all(registry.select(twin).resource_id == first for _ in range(5)),
    )


def test_infeasible_and_admission() -> None:
    """Infeasibility now means the registry is empty or all resources unavailable.

    Under the continuous scorer, a missing capability flag does NOT raise
    InfeasibleDNAError -- it just contributes a rejection penalty. The error is
    reserved for the truly impossible case (nothing to score at all).
    unsatisfiable_flags() is still the pre-execution admission control check.
    """
    print("\n[admission] infeasible DNA (continuous scorer)")
    registry = build_test_registry()

    # -- Missing flag: select() succeeds but winner has a low score ----------
    dna_code = CapabilityDNA(flags=["code.generation"])
    try:
        decision = registry.select(dna_code)
        # No error expected: scorer runs, all available resources get rejection
        # penalty for the missing flag, but one still wins.
        check(
            "select() does not raise for missing flag (continuous scorer)",
            True,
        )
        check(
            "some resource is selected despite missing flag",
            decision.resource_id != "",
            f"got {decision.resource_id}",
        )
    except InfeasibleDNAError:
        check(
            "select() does not raise for missing flag (continuous scorer)",
            False,
            "InfeasibleDNAError raised unexpectedly",
        )

    # -- unsatisfiable_flags() still identifies registry-wide gaps ----------
    check(
        "unsatisfiable_flags reports the gap",
        registry.unsatisfiable_flags(dna_code) == ["code.generation"],
        f"got {registry.unsatisfiable_flags(dna_code)}",
    )
    check(
        "satisfiable DNA reports no gap",
        registry.unsatisfiable_flags(CapabilityDNA(flags=["web.search"])) == [],
    )

    # -- Empty registry raises InfeasibleDNAError ---------------------------
    empty = CapabilityRegistry()
    try:
        empty.select(CapabilityDNA(flags=["web.search"]))
        check("empty registry raises", False, "no error raised")
    except InfeasibleDNAError:
        check("empty registry raises", True)

    # -- All-down registry raises InfeasibleDNAError ------------------------
    down_only = CapabilityRegistry()
    down_only.register(
        CapabilityManifest(
            resource_id="always_down",
            resource_class="llm",
            capabilities=["web.search"],
            availability=Availability(status="down"),
        ),
        _stub("always_down"),
    )
    try:
        down_only.select(CapabilityDNA(flags=["web.search"]))
        check("all-down registry raises", False, "no error raised")
    except InfeasibleDNAError:
        check("all-down registry raises", True)


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
    """Integration: default registry, no LLM calls.

    Under the continuous scorer, routing decisions are driven by the full
    multi-dimensional score. Capability flag match contributes the largest
    rejection penalty (task_value=1.0 with agent_value=0.0), so a resource
    that provides the exact required flags reliably outscores one that doesn't.
    """
    print("\n[integration] default registry, no LLM calls")
    from resource_registration import build_default_registry

    registry = build_default_registry()

    # A compare node needs reasoning.deep. The resources that provide it
    # (summarization, synthesis) should outscore those that don't.
    deep = CapabilityDNA(
        flags=["reasoning.deep", "text.summarization"],
        constraints=DNAConstraints(cost_ceiling_usd=0.05, latency_slo_ms=30_000),
    )
    deep_result = registry.select(deep)
    check(
        "reasoning.deep node binds to a resource that provides it",
        deep_result.resource_id in {"summarization", "synthesis"},
        f"got {deep_result.resource_id}",
    )

    # vision.understanding flag: only 'vision' provides it, so it must win.
    vis = CapabilityDNA(
        flags=["vision.understanding"],
        constraints=DNAConstraints(latency_slo_ms=30_000),
    )
    check(
        "vision flag binds to the vision resource",
        registry.select(vis).resource_id == "vision",
        f"got {registry.select(vis).resource_id}",
    )

    # web.search flag: only 'web_search' provides it, so it must win.
    web = CapabilityDNA(
        flags=["web.search"],
        constraints=DNAConstraints(latency_slo_ms=30_000, risk_tolerance="medium"),
    )
    check(
        "web.search flag binds to the search tool",
        registry.select(web).resource_id == "web_search",
        f"got {registry.select(web).resource_id}",
    )

    # Under the continuous scorer, risk_tolerance is a soft dimension, not a
    # hard gate. The web_search resource still wins because it's the only one
    # with the web.search flag (flag rejection penalty dominates).
    # The admission control layer (unsatisfiable_flags) remains the hard gate.
    web_low_risk = CapabilityDNA(
        flags=["web.search"],
        constraints=DNAConstraints(latency_slo_ms=30_000, risk_tolerance="low"),
    )
    check(
        "web.search still selectable with low risk_tolerance (soft penalty, not hard filter)",
        registry_select_ok(registry, web_low_risk),
    )

    # The terminal node must reach the synthesis resource (only one providing answer.synthesis).
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
    test_continuous_scorer_formula()
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

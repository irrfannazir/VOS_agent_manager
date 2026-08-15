"""Kernel-owned derivation of the DNA constraint segment.

The first live run made the case for this module. Asking the extractor LLM to
invent cost ceilings and latency SLOs produced `latency_slo_ms: 500` on every
node -- below the p95 of every registered resource -- so the feasibility filter
rejected everything and all nine nodes silently degraded to exact match. DNA
routing was decorative.

The model has no grounding for wall-clock or dollars. It *does* have grounding
for "how hard is this subtask", which is exactly the ordinal segment. So the
split is: the extractor emits flags + ordinals + confidence, and this policy
derives constraints from the job budget and the registry's actual resource
pool. Constraints become a property of the kernel's situation, not of the
model's imagination.
"""

from typing import TYPE_CHECKING

from models import CapabilityDNA, DNAConstraints

if TYPE_CHECKING:  # avoid an import cycle at runtime
    from capability_registry import CapabilityRegistry


# Flags that pull unvetted third-party content into the workflow. A node
# carrying one of these needs a risk tolerance high enough to admit the tools
# that fetch it, otherwise the filter rejects the only resource that can do
# the job -- which is what happened to node 'b' in the first live run.
_EXTERNAL_DATA_FLAGS = {"web.search", "tool.calling", "db.query", "sensor.reading"}

# Multiplier on the slowest feasible resource's p95. A node is allowed to be
# somewhat slower than the pool's worst case before the SLO bites; without
# slack, normal jitter reads as an SLO breach.
_LATENCY_SLACK = 1.5

# Ordinal demand maps into a quality floor over this band. A trivial node
# (demand 0) still needs a resource above 0.45; a maximally hard node (demand
# 1) needs 0.85. Going higher would exclude every resource in a small pool.
_MIN_QUALITY_FLOOR = 0.45
_MIN_QUALITY_CEILING = 0.85


class ConstraintPolicy:
    """Derives per-node DNA constraints from job-level budget and the resource pool.

    job_budget_usd is the whole task's spend envelope; it is divided across
    nodes rather than applied per node, so a 40-node plan automatically routes
    cheaper than a 3-node plan at the same budget. That division is the actual
    cost-control lever, and it is what a bandit would later learn to allocate
    non-uniformly.
    """

    def __init__(
        self,
        registry: "CapabilityRegistry",
        job_budget_usd: float = 0.50,
        job_latency_slo_ms: int | None = None,
    ) -> None:
        self.registry = registry
        self.job_budget_usd = job_budget_usd
        self.job_latency_slo_ms = job_latency_slo_ms

    # -- pool statistics ----------------------------------------------------

    def _pool_p95(self, flags: list[str]) -> int:
        """Worst p95 among resources that could plausibly serve these flags.

        Scoped to flag-relevant resources on purpose: a text node should not
        inherit the vision model's 15 s p95 just because it is registered.
        Falls back to the whole pool when nothing provides the flags, since the
        node is heading for a rejection anyway and a sane SLO keeps the
        rejection reason honest ("missing capability", not "SLO breach").
        """
        manifests = [
            m
            for m in self.registry.manifests()
            if not flags or set(flags).issubset(set(m.capabilities))
        ] or self.registry.manifests()

        if not manifests:
            return 30_000
        return max(m.latency_model.p95_ms for m in manifests)

    def _cheapest_cost(self, flags: list[str]) -> float:
        """Cost of the cheapest resource that can serve these flags."""
        costs = [
            m.cost_model.estimate_usd
            for m in self.registry.manifests()
            if not flags or set(flags).issubset(set(m.capabilities))
        ]
        return min(costs) if costs else 0.0

    # -- derivation ---------------------------------------------------------

    def derive(self, dna: CapabilityDNA, node_count: int) -> DNAConstraints:
        """Build the constraint segment for one node's DNA."""
        node_count = max(node_count, 1)

        # Cost: even share of the job budget, but never below what the cheapest
        # capable resource charges -- a ceiling that excludes every candidate is
        # a bug, not a budget.
        share = self.job_budget_usd / node_count
        cost_ceiling = max(share, self._cheapest_cost(dna.flags))

        # Latency: slack over the slowest capable resource, unless the caller
        # declared a job-level SLO, in which case divide it by the critical
        # path depth we approximate with node count.
        if self.job_latency_slo_ms is not None:
            latency_slo = max(
                int(self.job_latency_slo_ms / node_count), self._pool_p95(dna.flags)
            )
        else:
            latency_slo = int(self._pool_p95(dna.flags) * _LATENCY_SLACK)

        # Quality: linear in ordinal demand across a band that a small pool can
        # actually satisfy.
        demand = dna.ordinals.demand()
        min_quality = _MIN_QUALITY_FLOOR + demand * (
            _MIN_QUALITY_CEILING - _MIN_QUALITY_FLOOR
        )

        # Risk: a node that must reach outside the kernel has to tolerate the
        # risk class of the resources that do the reaching.
        needs_external = bool(_EXTERNAL_DATA_FLAGS.intersection(dna.flags))
        risk_tolerance = "high" if needs_external else "low"

        return DNAConstraints(
            cost_ceiling_usd=round(cost_ceiling, 6),
            latency_slo_ms=latency_slo,
            min_quality=round(min_quality, 4),
            risk_tolerance=risk_tolerance,
        )

    def apply(self, graph) -> None:
        """Overwrite the constraint segment on every node's DNA, in place."""
        node_count = len(graph.nodes)
        for node in graph.nodes:
            if node.dna is None:
                continue
            node.dna.constraints = self.derive(node.dna, node_count)

        print(
            f"[constraint-policy] budget ${self.job_budget_usd:.2f} across "
            f"{node_count} node(s) -> ${self.job_budget_usd / max(node_count, 1):.4f}"
            f"/node; constraints derived by kernel (not by the extractor)"
        )

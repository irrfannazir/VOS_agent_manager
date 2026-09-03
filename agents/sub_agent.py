from typing import List, Optional, Tuple

from aos_v0.capability_registry import CapabilityRegistry, InfeasibleDNAError
from aos_v0.failure_manager import FailureManager
from aos_v0.models import Node


class SubAgent:
    """Executes one DAG node. Decision-maker only -- all work goes to a Resource.

    Routing uses the continuous DNA scorer: every available resource is graded
    against the task's DNA and the highest-scoring one wins. The only hard gate
    is resource availability (status="up"). An infeasible DNA (empty registry
    or all resources unavailable) degrades to exact match rather than failing
    the node: DOC1 puts hard rejection in admission control, before execution,
    not here mid-wave.

    The actual invocation is handed to the FailureManager so every call runs
    inside the detect -> classify -> recover loop. The substitution ladder gets
    the scorer's ranked runners-up, which is why routing has to happen here
    rather than inside the failure manager.
    """

    def __init__(
        self,
        name: str,
        capability: str,
        registry: CapabilityRegistry,
        failure_manager: Optional[FailureManager] = None,
    ):
        self.name = name
        self.capability = capability
        self.registry = registry
        self.failure_manager = failure_manager or FailureManager(registry)

    def perform(self, node: Node) -> Node:
        node.status = "running"
        node.performed_by = self.name
        print(
            f"[{self.name}] performing '{_short(node.description)}' "
            f"(capability: {node.capability})"
        )

        fn, resource_id, substitutes = self._route(node)

        node.output = self.failure_manager.execute(
            node=node,
            primary_fn=fn,
            primary_resource_id=resource_id,
            candidate_ids=substitutes,
            log=print,
        )

        # execute() sets status to "degraded" when the ladder is exhausted; only
        # promote to done when it did not.
        if node.status != "degraded":
            node.status = "done"
        print(
            f"[{self.name}] {node.status} -> output length "
            f"{len(node.output or '')} chars"
        )
        return node

    # -- routing ------------------------------------------------------------

    def _route(self, node: Node) -> Tuple[object, str, List[str]]:
        """Return (run_fn, resource_id, ranked substitute ids)."""
        if not (node.dna and node.dna.flags):
            return self._route_exact(node, reason="no DNA flags")

        try:
            decision = self.registry.select(node.dna)
        except InfeasibleDNAError as exc:
            print(f"[{self.name}] DNA infeasible, trying relaxed routing: {exc}")
            return self._route_relaxed(node)

        node.bound_resource = decision.resource_id
        node.routing_mode = "dna"

        # Winner's detailed DNAScore (first in all_scores, ranked descending).
        winner_score = decision.all_scores[0]
        print(
            f"[{self.name}] DNA routing (continuous scorer): flags={node.dna.flags} -> "
            f"'{decision.resource_id}' (score={decision.score:.3f}, "
            f"accept={winner_score.acceptance_rate:.3f}, "
            f"reject={winner_score.rejection_rate:.3f}, "
            f"quality={decision.quality:.2f}, cost=${decision.cost_usd:.4f}, "
            f"p95={decision.latency_ms}ms)"
        )
        if decision.runner_up is not None:
            print(
                f"[{self.name}]   runner-up '{decision.runner_up}' "
                f"(margin={decision.runner_up_margin:.3f})"
            )

        # Substitutes = all scored resources except the winner, in score order.
        substitutes = [
            s.resource_id
            for s in decision.all_scores
            if s.resource_id != decision.resource_id
        ]

        # When no other resource was scored (winner is the only available one),
        # offer partial-flag-overlap resources as fallback substitutes so the
        # FailureManager has something to try on resource.outage.
        if not substitutes:
            node_flags = set(node.dna.flags)
            substitutes = [
                m.resource_id
                for m in self.registry.manifests()
                if m.resource_id != decision.resource_id
                and node_flags & set(m.capabilities)
            ]

        return self.registry.run_fn(decision.resource_id), decision.resource_id, substitutes


    def _route_relaxed(self, node: Node) -> Tuple[object, str, List[str]]:
        """Dynamic fallback: find ANY resource that provides at least one DNA flag.

        Instead of falling back to exact-match on the manager's coarse capability
        string (which may be completely wrong, e.g. 'web_search' for an audio
        task), we search the registry for resources that overlap with the DNA
        flags. This keeps routing capability-driven even when strict feasibility
        fails.

        Partial matches are marked degraded so the synthesis node can report
        capability-specific failures rather than treating them as normal success.
        """
        node_flags = set(node.dna.flags) if node.dna else set()
        best_rid = None
        best_overlap = 0
        all_manifests = self.registry.manifests()

        for m in all_manifests:
            m_caps = set(m.capabilities)
            overlap = len(node_flags & m_caps)
            if overlap > best_overlap:
                best_overlap = overlap
                best_rid = m.resource_id

        if best_rid and best_overlap > 0:
            fn = self.registry.run_fn(best_rid)
            node.bound_resource = best_rid
            node.routing_mode = "relaxed"
            # Mark degraded so synthesis reports the capability gap.
            node.status = "degraded"
            print(
                f"[{self.name}] RELAXED routing: {best_overlap} flag(s) overlap "
                f"-> '{best_rid}' (partial match, marked degraded)"
            )
            # Offer other overlapping resources as substitutes.
            substitutes = [
                m.resource_id for m in all_manifests
                if m.resource_id != best_rid and node_flags & set(m.capabilities)
            ]
            return fn, best_rid, substitutes

        # Last resort: exact-match on the capability string.
        print(f"[{self.name}] no flag overlap found, falling back to exact match")
        return self._route_exact(node, reason="no flag overlap")

    def _route_exact(self, node: Node, reason: str) -> Tuple[object, str, List[str]]:
        fn = self.registry.find_by_capability(node.capability)
        node.bound_resource = node.capability
        node.routing_mode = "exact"
        print(f"[{self.name}] exact-match routing on '{node.capability}' ({reason})")

        # Without DNA there is no scored candidate list, so offer any resource
        # that provides the same coarse capability as a substitute. Better than
        # no recovery path at all, weaker than the DNA-routed ladder.
        substitutes = [
            m.resource_id
            for m in self.registry.manifests()
            if m.resource_id != node.capability
            and self._same_family(m.resource_id, node.capability)
        ]
        return fn, node.capability, substitutes

    @staticmethod
    def _same_family(resource_id: str, capability: str) -> bool:
        """Crude sibling test for the no-DNA path (quick_summarization <-> summarization)."""
        return capability in resource_id or resource_id in capability


def _short(text: str, limit: int = 70) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."

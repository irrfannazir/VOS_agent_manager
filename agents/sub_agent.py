from typing import List, Optional, Tuple

from aos_v0.capability_registry import CapabilityRegistry, InfeasibleDNAError
from aos_v0.failure_manager import FailureManager
from aos_v0.models import Node


class SubAgent:
    """Executes one DAG node. Decision-maker only -- all work goes to a Resource.

    Routing is the M5 two-stage decision when the node carries Capability DNA,
    and the exact-match lookup otherwise. An infeasible DNA degrades to exact
    match rather than failing the node: DOC1 puts hard rejection in admission
    control, before execution, not here mid-wave.

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
            print(f"[{self.name}] DNA infeasible, degrading to exact match: {exc}")
            return self._route_exact(node, reason="DNA infeasible")

        node.bound_resource = decision.resource_id
        node.routing_mode = "dna"
        print(
            f"[{self.name}] DNA routing: flags={node.dna.flags} -> "
            f"'{decision.resource_id}' (score={decision.score:.3f}, "
            f"quality={decision.quality:.2f}, cost=${decision.cost_usd:.4f}, "
            f"p95={decision.latency_ms}ms)"
        )
        if decision.runner_up is not None:
            print(
                f"[{self.name}]   runner-up '{decision.runner_up}' "
                f"(margin={decision.runner_up_margin:.3f})"
            )

        substitutes = [
            c.resource_id
            for c in decision.candidates
            if c.resource_id != decision.resource_id
        ]
        return self.registry.run_fn(decision.resource_id), decision.resource_id, substitutes

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

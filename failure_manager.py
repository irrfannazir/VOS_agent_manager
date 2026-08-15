"""M8-lite -- Cognitive Failure Manager.

The closed loop DOC1 asks for, at the scale this single-process kernel needs:

  detection ensemble -> classification -> recovery policy selection

The motivating case is real. In the first live run, node 'd' ("results of all
matches of Argentina") returned 27 characters meaning "not found", nothing
noticed, and the final answer told the user their question could not be
answered. An exception-only retry would not have caught it either: the node
succeeded, it just succeeded at producing nothing. That is precisely the
"cognitive failure" distinction -- a deviation detected at the semantic level
rather than an exception.

Recovery is a policy table, not a reflex: the failure is classified first, and
the class picks the strategy ladder. Strategies escalate, so a transient blip
costs one cheap retry while a genuinely dry query moves to a different resource
instead of asking the same thing five times.
"""

import re
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from capability_registry import CapabilityRegistry
from models import Node


# --- failure taxonomy (MAST-extended subset, DOC1 M8 stage 2) --------------

CLASS_RESOURCE_OUTAGE = "resource.outage"
CLASS_RESOURCE_DEGRADED = "resource.degraded"
CLASS_TOOL_EMPTY_RESULT = "tool.empty_result"
CLASS_TOOL_OUTPUT_CORRUPT = "tool.output_corrupt"
CLASS_REASONING_REFUSAL = "reasoning.refusal"

# --- recovery strategies (DOC1 M8 stage 3) ---------------------------------

STRATEGY_RETRY_SAME = "retry_same"
STRATEGY_RETRY_WITH_FEEDBACK = "retry_with_feedback"
STRATEGY_RESOURCE_SUBSTITUTION = "resource_substitution"

# Hand-crafted taxonomy -> strategy ladder. DOC1 ships this table first and
# replaces it with a bandit later; the ladder order IS the policy. Each class
# is tried in order and stops at the first strategy that produces healthy
# output.
RECOVERY_TABLE: dict[str, List[str]] = {
    # A thrown exception is usually transient (rate limit, socket) -- one cheap
    # retry, then move off the resource entirely.
    CLASS_RESOURCE_OUTAGE: [STRATEGY_RETRY_SAME, STRATEGY_RESOURCE_SUBSTITUTION],
    # Output arrived but is too thin to use. Retrying identically would return
    # the same thin output, so reformulate first.
    CLASS_TOOL_EMPTY_RESULT: [
        STRATEGY_RETRY_WITH_FEEDBACK,
        STRATEGY_RESOURCE_SUBSTITUTION,
    ],
    CLASS_RESOURCE_DEGRADED: [
        STRATEGY_RETRY_WITH_FEEDBACK,
        STRATEGY_RESOURCE_SUBSTITUTION,
    ],
    # Structurally broken output: a different resource is more likely to help
    # than another attempt at the same one.
    CLASS_TOOL_OUTPUT_CORRUPT: [
        STRATEGY_RESOURCE_SUBSTITUTION,
        STRATEGY_RETRY_WITH_FEEDBACK,
    ],
    # A model declining the task will decline it again; only rephrasing helps.
    CLASS_REASONING_REFUSAL: [STRATEGY_RETRY_WITH_FEEDBACK],
}

# Detector thresholds. Deliberately conservative -- DOC1 R3 flags false-positive
# recovery as the main cost risk, so a node must be clearly broken, not merely
# terse, before recovery spends money on it.
_MIN_USEFUL_CHARS = 60

_EMPTY_RESULT_PATTERNS = [
    re.compile(r"^\s*NO_DATA:", re.IGNORECASE),
    re.compile(r"\bnot found in search results\b", re.IGNORECASE),
    re.compile(r"\bno (?:web )?(?:search )?results? (?:were )?found\b", re.IGNORECASE),
    re.compile(r"\bno (?:relevant |available )?information (?:is |was )?available\b", re.IGNORECASE),
    re.compile(r"\bunable to (?:find|retrieve|locate)\b", re.IGNORECASE),
]

_CORRUPT_PATTERNS = [
    re.compile(r"^\s*\[Search error:", re.IGNORECASE),
    re.compile(r"\brate.?limit\b", re.IGNORECASE),
]

_REFUSAL_PATTERNS = [
    re.compile(r"\bI (?:cannot|can't|am unable to) (?:assist|help|provide)\b", re.IGNORECASE),
    re.compile(r"\bas an AI language model\b", re.IGNORECASE),
]


@dataclass
class Detection:
    """One detector firing, with the class it implies."""

    failure_class: str
    symptom: str


@dataclass
class RecoveryAttempt:
    strategy: str
    resource_id: str
    succeeded: bool
    detail: str = ""


@dataclass
class NodeOutcome:
    """Per-node record. Feeds the Experience Record the Learning Manager writes."""

    node_id: str
    detections: List[Detection] = field(default_factory=list)
    attempts: List[RecoveryAttempt] = field(default_factory=list)
    recovered: bool = False
    degraded: bool = False
    elapsed_ms: int = 0

    def summary(self) -> str:
        if not self.detections:
            return "healthy"
        classes = ", ".join(d.failure_class for d in self.detections)
        if self.recovered:
            fired = self.attempts[-1].strategy if self.attempts else "?"
            return f"recovered from [{classes}] via {fired}"
        return f"DEGRADED, unrecovered [{classes}]"


class FailureManager:
    """Detects, classifies and recovers cognitive failures around a node call.

    max_attempts bounds the ladder so a permanently dry query costs a fixed,
    known amount rather than looping. Nodes that exhaust the ladder are marked
    degraded and their output is replaced with an explicit gap marker, so the
    synthesis node reports the gap honestly instead of inheriting noise.
    """

    def __init__(self, registry: CapabilityRegistry, max_attempts: int = 2):
        self.registry = registry
        self.max_attempts = max_attempts
        self.outcomes: dict[str, NodeOutcome] = {}

    # -- stage 1: detection ensemble ---------------------------------------

    def detect(self, node: Node, output: Optional[str], error: Optional[Exception]) -> List[Detection]:
        """Run every detector. Returns all firings, empty list means healthy."""
        detections: List[Detection] = []

        if error is not None:
            return [
                Detection(
                    CLASS_RESOURCE_OUTAGE,
                    f"{type(error).__name__}: {error}",
                )
            ]

        if output is None or not output.strip():
            return [Detection(CLASS_TOOL_EMPTY_RESULT, "empty output")]

        for pattern in _CORRUPT_PATTERNS:
            if pattern.search(output):
                detections.append(
                    Detection(CLASS_TOOL_OUTPUT_CORRUPT, f"matched {pattern.pattern!r}")
                )
                break

        for pattern in _REFUSAL_PATTERNS:
            if pattern.search(output):
                detections.append(
                    Detection(CLASS_REASONING_REFUSAL, f"matched {pattern.pattern!r}")
                )
                break

        for pattern in _EMPTY_RESULT_PATTERNS:
            if pattern.search(output):
                detections.append(
                    Detection(CLASS_TOOL_EMPTY_RESULT, f"matched {pattern.pattern!r}")
                )
                break

        # Length check runs last and only if nothing else fired, so a short but
        # legitimately terse answer is not double-reported.
        if not detections and len(output.strip()) < _MIN_USEFUL_CHARS:
            detections.append(
                Detection(
                    CLASS_RESOURCE_DEGRADED,
                    f"output only {len(output.strip())} chars "
                    f"(< {_MIN_USEFUL_CHARS})",
                )
            )

        return detections

    # -- stage 2: classification -------------------------------------------

    @staticmethod
    def classify(detections: List[Detection]) -> Optional[str]:
        """Pick the governing class when several detectors fire.

        Ordered by how much the class constrains recovery: an outage rules out
        the resource entirely, corruption rules out its output, and thinness is
        the weakest signal. Taking the strongest avoids treating a crashed
        resource as merely terse.
        """
        if not detections:
            return None
        priority = [
            CLASS_RESOURCE_OUTAGE,
            CLASS_TOOL_OUTPUT_CORRUPT,
            CLASS_REASONING_REFUSAL,
            CLASS_TOOL_EMPTY_RESULT,
            CLASS_RESOURCE_DEGRADED,
        ]
        fired = {d.failure_class for d in detections}
        for cls in priority:
            if cls in fired:
                return cls
        return detections[0].failure_class

    # -- stage 3: recovery --------------------------------------------------

    def execute(
        self,
        node: Node,
        primary_fn: Callable,
        primary_resource_id: str,
        candidate_ids: List[str],
        log: Callable[[str], None] = print,
    ) -> str:
        """Run the node under the closed loop and return its final output.

        `candidate_ids` is the scorer's ranked feasible list minus the winner --
        substitution walks it in score order, so the substitute is the
        runner-up the Pareto scorer already vetted rather than an arbitrary
        resource that happens to share a capability string.
        """
        outcome = NodeOutcome(node_id=node.id)
        self.outcomes[node.id] = outcome
        started = time.monotonic()

        output, error = self._invoke(primary_fn, node.input, node.description)
        detections = self.detect(node, output, error)
        outcome.detections = detections

        if not detections:
            outcome.elapsed_ms = int((time.monotonic() - started) * 1000)
            return output

        failure_class = self.classify(detections)
        symptom = detections[0].symptom
        log(
            f"[failure-manager] node '{node.id}': detected {failure_class} "
            f"({symptom})"
        )

        ladder = RECOVERY_TABLE.get(failure_class, [STRATEGY_RETRY_SAME])
        remaining = list(candidate_ids)
        best_output = output

        for strategy in ladder[: self.max_attempts]:
            log(f"[failure-manager] node '{node.id}': recovery -> {strategy}")

            if strategy == STRATEGY_RETRY_SAME:
                attempt_fn, attempt_rid = primary_fn, primary_resource_id
                instruction = node.description

            elif strategy == STRATEGY_RETRY_WITH_FEEDBACK:
                attempt_fn, attempt_rid = primary_fn, primary_resource_id
                instruction = self._reformulate(node.description, failure_class)
                log(f"[failure-manager] node '{node.id}': reformulated -> {instruction!r}")

            elif strategy == STRATEGY_RESOURCE_SUBSTITUTION:
                if not remaining:
                    log(
                        f"[failure-manager] node '{node.id}': no substitute "
                        f"resource available, skipping"
                    )
                    outcome.attempts.append(
                        RecoveryAttempt(strategy, "-", False, "no candidates")
                    )
                    continue
                attempt_rid = remaining.pop(0)
                attempt_fn = self.registry.run_fn(attempt_rid)
                instruction = node.description
                log(f"[failure-manager] node '{node.id}': substituting -> '{attempt_rid}'")

            else:  # pragma: no cover -- table and constants are in sync
                continue

            retry_output, retry_error = self._invoke(attempt_fn, node.input, instruction)
            retry_detections = self.detect(node, retry_output, retry_error)

            if not retry_detections:
                outcome.attempts.append(RecoveryAttempt(strategy, attempt_rid, True))
                outcome.recovered = True
                outcome.elapsed_ms = int((time.monotonic() - started) * 1000)
                node.bound_resource = attempt_rid
                log(
                    f"[failure-manager] node '{node.id}': RECOVERED via "
                    f"{strategy} on '{attempt_rid}'"
                )
                return retry_output

            outcome.attempts.append(
                RecoveryAttempt(
                    strategy, attempt_rid, False, retry_detections[0].symptom
                )
            )
            # Keep the longest output seen: even a failed attempt may carry
            # more usable signal than the original, and discarding it would
            # make recovery strictly worse than not trying.
            if retry_output and len(retry_output) > len(best_output or ""):
                best_output = retry_output

        outcome.degraded = True
        outcome.elapsed_ms = int((time.monotonic() - started) * 1000)
        node.status = "degraded"
        log(
            f"[failure-manager] node '{node.id}': UNRECOVERED after "
            f"{len(outcome.attempts)} attempt(s) -- marking degraded"
        )
        return self._gap_marker(node, failure_class, best_output)

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _invoke(fn: Callable, node_input, instruction):
        try:
            return fn(node_input, instruction=instruction), None
        except Exception as exc:  # noqa: BLE001 -- classification is the point
            return None, exc

    @staticmethod
    def _reformulate(description: str, failure_class: str) -> str:
        """Rewrite the instruction so a retry is not a verbatim repeat.

        Deliberately mechanical rather than another LLM call: the reformulation
        must be cheap and deterministic, or recovery cost becomes unbounded and
        the seeded-replay determinism audit stops holding.
        """
        if failure_class == CLASS_REASONING_REFUSAL:
            return (
                f"{description}\n\nReport only publicly available factual "
                f"information. If some part cannot be answered, state which "
                f"part and continue with the rest."
            )
        return (
            f"{description}\n\nThe first attempt returned no usable data. "
            f"Broaden the query: drop qualifiers, try the entity's common name "
            f"and alternative phrasings, and return whatever partial facts, "
            f"dates and figures you can find rather than nothing."
        )

    @staticmethod
    def _gap_marker(node: Node, failure_class: str, best_output: Optional[str]) -> str:
        """Explicit, machine-readable gap text for downstream nodes.

        The synthesis node is instructed to report gaps honestly, so handing it
        a labelled gap produces "no match-by-match results were found" instead
        of the silent hole the first run produced.
        """
        marker = (
            f"[UNAVAILABLE — node '{node.id}' failed with {failure_class} after "
            f"recovery attempts. Requested: {node.description}]"
        )
        if best_output and best_output.strip():
            return f"{marker}\nPartial data recovered:\n{best_output.strip()}"
        return marker

    # -- reporting ----------------------------------------------------------

    def report(self) -> str:
        lines = ["=== FAULT REPORT ==="]
        faults = [o for o in self.outcomes.values() if o.detections]
        if not faults:
            lines.append("  no failures detected")
            return "\n".join(lines)

        recovered = sum(1 for o in faults if o.recovered)
        degraded = sum(1 for o in faults if o.degraded)
        for outcome in faults:
            lines.append(f"  {outcome.node_id:<8} {outcome.summary()}")
            for attempt in outcome.attempts:
                status = "ok" if attempt.succeeded else "failed"
                detail = f" ({attempt.detail})" if attempt.detail else ""
                lines.append(
                    f"      - {attempt.strategy} on '{attempt.resource_id}': "
                    f"{status}{detail}"
                )
        lines.append(
            f"  totals: {len(faults)} node(s) faulted, {recovered} recovered, "
            f"{degraded} degraded"
        )
        return "\n".join(lines)

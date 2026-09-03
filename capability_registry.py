"""Capability Registry: resource manifests + continuous DNA-score scheduling.

DOC1 5.1 defines what a resource publishes (CapabilityManifest). The original
M5 two-stage binding (feasibility filter → Pareto scorer) has been replaced by
a single *continuous* DNA scoring pass:

    score = acceptance_rate * PESSIMISING_FACTOR
          - rejection_rate  * OPTIMISING_FACTOR

For each capability dimension (required flags, ordinal axes, cost, latency,
quality):

    if agent_value > task_value:
        acceptance_rate += ((agent_value - task_value) / agent_value) * weight
    else:
        rejection_rate  += ((task_value - agent_value)
                            / (1 - agent_value)) * weight

Division-by-zero guards are applied throughout. The resource with the highest
score across all dimensions wins for each task node.

The only hard gate that survives is *availability*: a resource that is not
status="up" is excluded before scoring begins (liveness, not capability).
InfeasibleDNAError is kept and raised only when the registry is empty or every
resource is unavailable, so the caller always gets a meaningful error rather
than an empty score list.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from models import CAPABILITY_FLAGS, CapabilityDNA, risk_rank


ResourceClass = Literal[
    "llm",
    "vlm",
    "asr",
    "tts",
    "tool",
    "api",
    "database",
    "sensor",
    "robot",
    "embedder",
    "reranker",
    # Multimodal classes added with the multimodal-catalog expansion. `audio`
    # covers audio-understanding / audio-classification models that are not
    # speech-to-text (ASR); `image` covers pure vision models that are not
    # vision-language (VLM). Like embedder/reranker, these are informational
    # class tags -- scheduling still routes on capability flags only.
    "audio",
    "image",
]


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class InfeasibleDNAError(RuntimeError):
    """No registered resource is available to serve a subtask's DNA.

    Raised only when the registry is empty or every resource is unavailable
    (status != 'up'). A resource that merely scores poorly is still eligible
    -- the continuous scorer rewards over-qualification and penalises gaps,
    but never hard-excludes based on capability mismatch alone.

    Carries a structured reason per resource so callers can report *why*
    nothing was available rather than just "no match".
    """

    def __init__(self, dna: CapabilityDNA, rejections: Dict[str, str]):
        self.dna = dna
        self.rejections = rejections
        detail = "; ".join(f"{name}: {why}" for name, why in sorted(rejections.items()))
        super().__init__(
            f"[capability-registry] no available resource for DNA "
            f"flags={dna.flags} ordinals={dna.ordinals.model_dump()} "
            f"constraints={dna.constraints.model_dump()}. Rejections -> "
            f"{detail or '(no resources registered)'}"
        )


def missing_flags(dna: CapabilityDNA, provided: List[str]) -> List[str]:
    """Flags the subtask needs that nothing in `provided` supplies."""
    have = set(provided)
    return [f for f in dna.flags if f not in have]


# ---------------------------------------------------------------------------
# CapabilityManifest (DOC1 5.1)
# ---------------------------------------------------------------------------


class IOSchema(BaseModel):
    type: Literal["text", "image", "audio", "structured"] = "text"
    format: str = "plain"


class CostModel(BaseModel):
    unit: Literal["per_1k_tokens", "per_call"] = "per_call"
    estimate_usd: float = Field(default=0.0, ge=0.0)


class LatencyModel(BaseModel):
    p50_ms: int = Field(default=1000, ge=0)
    p95_ms: int = Field(default=4000, ge=0)


class Availability(BaseModel):
    status: Literal["up", "degraded", "down"] = "up"
    rate_limit_rpm: int = Field(default=60, ge=0)


class CapabilityManifest(BaseModel):
    """What every Resource publishes via describe(). DOC1 5.1.

    No Callable field on purpose: the manifest stays a pure, serialisable
    declaration (it goes straight into traces and plan.md), while the run
    function lives in a parallel dict on the registry.
    """

    resource_id: str
    resource_class: ResourceClass
    capabilities: List[str]
    input_schema: IOSchema = Field(default_factory=IOSchema)
    output_schema: IOSchema = Field(default_factory=IOSchema)
    cost_model: CostModel = Field(default_factory=CostModel)
    latency_model: LatencyModel = Field(default_factory=LatencyModel)
    quality_priors: Dict[str, float] = Field(default_factory=dict)
    availability: Availability = Field(default_factory=Availability)
    risk_class: Literal["low", "medium", "high"] = "low"
    # Provider-specific audit metadata (e.g. {"provider": "huggingface",
    # "model": "..."}). Deliberately free-form and optional: scheduling routes
    # on `capabilities` only, so the core stays provider-agnostic and this is
    # trace/plan data, never decision logic.
    metadata: Dict[str, str] = Field(default_factory=dict)

    @field_validator("capabilities")
    @classmethod
    def _capabilities_in_vocabulary(cls, caps: List[str]) -> List[str]:
        unknown = [c for c in caps if c not in CAPABILITY_FLAGS]
        if unknown:
            raise ValueError(
                f"unknown capability flag(s) {unknown}; "
                f"vocabulary is {CAPABILITY_FLAGS}"
            )
        return caps

    @field_validator("quality_priors")
    @classmethod
    def _priors_in_vocabulary(cls, priors: Dict[str, float]) -> Dict[str, float]:
        unknown = [k for k in priors if k not in CAPABILITY_FLAGS]
        if unknown:
            raise ValueError(f"quality_priors keyed by unknown flag(s) {unknown}")
        bad = {k: v for k, v in priors.items() if not 0.0 <= v <= 1.0}
        if bad:
            raise ValueError(f"quality_priors must be in [0,1], got {bad}")
        return priors

    def quality_for(self, flags: List[str]) -> float:
        """Mean prior across the requested flags.

        A flag the resource covers but has no measured prior for defaults to
        0.5 -- neutral, so an unmeasured resource is neither auto-selected nor
        auto-excluded. With no flags requested there is nothing to average, so
        fall back to the resource's overall mean (or 0.5 if it has no priors).
        """
        if not flags:
            if not self.quality_priors:
                return 0.5
            return sum(self.quality_priors.values()) / len(self.quality_priors)
        return sum(self.quality_priors.get(f, 0.5) for f in flags) / len(flags)


# ---------------------------------------------------------------------------
# Continuous DNA scorer — data types
# ---------------------------------------------------------------------------


@dataclass
class DimensionWeights:
    """Per-dimension weights fed into the continuous DNA scorer.

    Each weight scales the acceptance / rejection contribution of that
    dimension in the final score. All weights default to 1.0 (equal
    importance). Tune these to bias routing toward capability match over
    cost, or toward latency over quality, etc.

    Dimension map
    -------------
    flag_match   : whether the resource provides each required capability flag
    reasoning    : reasoning_depth ordinal vs reasoning quality prior
    planning     : planning_horizon ordinal vs planning.decomposition prior
    tool         : tool_complexity ordinal vs tool.calling prior
    cost         : cost_ceiling vs resource cost estimate
    latency      : latency_slo vs resource p95 latency
    quality      : effective min_quality vs mean quality prior across flags
    """

    flag_match: float = 1.0
    reasoning: float = 1.0
    planning: float = 1.0
    tool: float = 1.0
    cost: float = 1.0
    latency: float = 1.0
    quality: float = 1.0


@dataclass
class DimensionDetail:
    """Contribution of one dimension to an agent's DNA score."""

    name: str
    agent_value: float
    task_value: float
    weight: float
    contribution: float   # positive = acceptance, negative = rejection
    accepted: bool        # True if agent_value > task_value


@dataclass
class DNAScore:
    """Full audit record of one continuous DNA scoring decision.

    acceptance_rate and rejection_rate are the raw accumulators before the
    PESSIMISING / OPTIMISING constants are applied, so the trace shows both
    the raw signal and how the constants shaped the final score.
    """

    resource_id: str
    score: float
    acceptance_rate: float
    rejection_rate: float
    dimensions: List[DimensionDetail] = field(default_factory=list)

    # Kept for API compatibility with code that reads .quality / .cost_usd /
    # .latency_ms from the old ScheduleDecision / Candidate types.
    quality: float = 0.0
    cost_usd: float = 0.0
    latency_ms: int = 0


@dataclass
class SelectionResult:
    """Return type of CapabilityRegistry.select(): winner + full ranked list."""

    resource_id: str
    score: float
    quality: float
    cost_usd: float
    latency_ms: int
    all_scores: List[DNAScore]

    # Runner-up fields kept for backward compat with sub_agent.py log lines.
    @property
    def runner_up(self) -> Optional[str]:
        if len(self.all_scores) > 1:
            return self.all_scores[1].resource_id
        return None

    @property
    def runner_up_margin(self) -> Optional[float]:
        if len(self.all_scores) > 1:
            return self.score - self.all_scores[1].score
        return None

    @property
    def candidates(self) -> List[DNAScore]:
        """Expose ranked scores as 'candidates' for code that iterates them."""
        return self.all_scores


# Backward compat aliases — old names still importable.
Candidate = DNAScore
ScheduleDecision = SelectionResult


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Default scoring constants. Symmetric (1.0 / 1.0) means acceptance and
# rejection contributions are on equal footing. Raise PESSIMISING_FACTOR to
# reward over-qualified agents more; raise OPTIMISING_FACTOR to penalise
# under-qualified agents more.
_DEFAULT_PESSIMISING_FACTOR: float = 1.0
_DEFAULT_OPTIMISING_FACTOR: float = 1.0

# Ordinal axes are on a 0-4 integer scale; normalise to [0, 1] before
# comparison so they are commensurable with the quality/cost/latency values
# which are already in [0, 1].
_ORDINAL_MAX: float = 4.0

# Map from ordinal field name to the manifest quality-prior flag that best
# represents the agent's strength on that axis.
_ORDINAL_TO_FLAG: Dict[str, str] = {
    "reasoning_depth": "reasoning.deep",
    "planning_horizon": "planning.decomposition",
    "tool_complexity": "tool.calling",
}


class CapabilityRegistry:
    """Live manifest table + continuous DNA-score binding.

    pessimising_factor / optimising_factor are the scoring constants applied
    to the acceptance and rejection accumulators respectively:

        score = acceptance_rate * pessimising_factor
              - rejection_rate  * optimising_factor

    weights is a DimensionWeights instance controlling per-dimension
    importance. All default to 1.0 (equal weight).
    """

    def __init__(
        self,
        pessimising_factor: float = _DEFAULT_PESSIMISING_FACTOR,
        optimising_factor: float = _DEFAULT_OPTIMISING_FACTOR,
        weights: Optional[DimensionWeights] = None,
        # Legacy Pareto params kept so callers that pass them don't break.
        lambda_cost: float = 0.3,
        mu_latency: float = 0.2,
    ) -> None:
        self._manifests: Dict[str, CapabilityManifest] = {}
        self._run_fns: Dict[str, Callable] = {}
        self.pessimising_factor = pessimising_factor
        self.optimising_factor = optimising_factor
        self.weights = weights or DimensionWeights()
        # Legacy attributes retained for any code that reads them directly.
        self.lambda_cost = lambda_cost
        self.mu_latency = mu_latency

    # -- registration -------------------------------------------------------

    def register(self, manifest: CapabilityManifest, run_fn: Callable) -> None:
        self._manifests[manifest.resource_id] = manifest
        self._run_fns[manifest.resource_id] = run_fn

    def describe(self, resource_id: str) -> CapabilityManifest:
        return self._manifests[resource_id]

    def manifests(self) -> List[CapabilityManifest]:
        return list(self._manifests.values())

    def run_fn(self, resource_id: str) -> Callable:
        return self._run_fns[resource_id]

    def provided_flags(self) -> List[str]:
        """Union of every capability any registered resource provides."""
        provided: set[str] = set()
        for manifest in self._manifests.values():
            provided.update(manifest.capabilities)
        return sorted(provided)

    # -- M1: plain lookups --------------------------------------------------

    def find(self, flags: List[str]) -> List[str]:
        """Resource ids whose capabilities are a superset of `flags` (DOC1 M1)."""
        wanted = set(flags)
        return sorted(
            rid
            for rid, manifest in self._manifests.items()
            if wanted.issubset(set(manifest.capabilities))
        )

    def find_by_capability(self, capability: str) -> Callable:
        """Exact-match fallback, mirroring v0.0.2's CAPABILITY_MAP lookup.

        Used only for nodes with no DNA. Resources are registered under ids
        matching the coarse capability strings, so this is a key lookup.
        """
        if capability not in self._run_fns:
            available = ", ".join(sorted(self._run_fns)) or "(none registered)"
            raise KeyError(
                f"[capability-registry] no resource registered for capability "
                f"'{capability}'. Available: {available}"
            )
        return self._run_fns[capability]

    # -- admission control seed (DOC1 M2) -----------------------------------

    def unsatisfiable_flags(self, dna: CapabilityDNA) -> List[str]:
        """Flags no registered resource provides at all.

        Distinct from the scoring pass: this is the hard, pre-execution check
        the Task Manager uses to reject a plan by *naming the missing
        capability*, rather than a soft scoring miss that a partial-match
        resource might partially cover.
        """
        return missing_flags(dna, self.provided_flags())

    # -- continuous DNA scorer ----------------------------------------------

    @staticmethod
    def _score_dimension(
        agent_value: float,
        task_value: float,
        weight: float,
    ) -> Tuple[float, float]:
        """Compute (acceptance_contribution, rejection_contribution) for one dimension.

        Formula (per the design spec):
            if agent_value > task_value:
                acceptance += ((agent_value - task_value) / agent_value) * weight
            else:
                rejection  += ((task_value - agent_value) / (1 - agent_value)) * weight

        Division-by-zero guards:
            • agent_value == 0 in the acceptance branch → contribution = 0.0
              (agent has nothing to offer; no gain, no penalty — handled by
              the else branch since 0 > task_value is always False for
              task_value >= 0, but guarded defensively anyway).
            • agent_value == 1 in the rejection branch → divisor is zero;
              cap contribution at weight (maximum penalty) — agent claims
              perfection but still falls short of the task requirement.
        """
        if agent_value > task_value:
            # Guard: agent_value == 0 → no headroom to measure gain.
            if agent_value == 0.0:
                return 0.0, 0.0
            acceptance = ((agent_value - task_value) / agent_value) * weight
            return acceptance, 0.0
        else:
            # agent_value <= task_value → under-qualified on this dimension.
            denominator = 1.0 - agent_value
            if denominator == 0.0:
                # agent_value == 1 but still <= task_value (only when task_value
                # is also 1 exactly); contribution is effectively zero gap.
                rejection = 0.0
            else:
                rejection = ((task_value - agent_value) / denominator) * weight
            return 0.0, rejection

    def score_against_dna(
        self, manifest: CapabilityManifest, dna: CapabilityDNA
    ) -> DNAScore:
        """Grade one resource against a task's DNA using the continuous scorer.

        Returns a DNAScore with the final score plus per-dimension audit trail.
        The score is:

            score = acceptance_rate * pessimising_factor
                  - rejection_rate  * optimising_factor

        Dimensions evaluated
        --------------------
        1. Flag match  — for each required flag: 1.0 if provided, 0.0 if not.
        2. Reasoning   — reasoning_depth ordinal (normalised) vs quality prior
                         on reasoning.deep (or reasoning.shallow as fallback).
        3. Planning    — planning_horizon ordinal vs planning.decomposition prior.
        4. Tool        — tool_complexity ordinal vs tool.calling prior.
        5. Cost        — task cost ceiling vs resource cost, normalised to [0,1].
        6. Latency     — task latency SLO vs resource p95, normalised to [0,1].
        7. Quality     — effective min_quality vs mean quality prior across flags.
        """
        w = self.weights
        dimensions: List[DimensionDetail] = []
        acceptance_rate: float = 0.0
        rejection_rate: float = 0.0

        def _record(name, agent_val, task_val, weight):
            nonlocal acceptance_rate, rejection_rate
            acc, rej = self._score_dimension(agent_val, task_val, weight)
            acceptance_rate += acc
            rejection_rate += rej
            dimensions.append(DimensionDetail(
                name=name,
                agent_value=agent_val,
                task_value=task_val,
                weight=weight,
                contribution=acc - rej,
                accepted=agent_val > task_val,
            ))

        agent_caps = set(manifest.capabilities)

        # 1. Required capability flag match
        for flag in dna.flags:
            agent_val = 1.0 if flag in agent_caps else 0.0
            # task always requires the flag fully (1.0)
            _record(f"flag:{flag}", agent_val, 1.0, w.flag_match)

        # 2. Ordinal axes: reasoning, planning, tool
        for ordinal_field, proxy_flag, dim_name, dim_weight in [
            ("reasoning_depth", "reasoning.deep",         "reasoning", w.reasoning),
            ("planning_horizon", "planning.decomposition", "planning",  w.planning),
            ("tool_complexity",  "tool.calling",           "tool",      w.tool),
        ]:
            ordinal_val: int = getattr(dna.ordinals, ordinal_field, 0)
            task_val = ordinal_val / _ORDINAL_MAX  # normalise 0-4 to 0-1

            # Agent value: quality prior on the proxy flag, or 0.5 neutral if
            # the agent doesn't declare a prior for that flag.
            agent_val = manifest.quality_priors.get(proxy_flag, 0.5)
            _record(dim_name, agent_val, task_val, dim_weight)

        # 3. Cost — cheaper resource = higher agent_value.
        #
        #    agent_value = 1 - cost/ceiling : ranges 0 (at ceiling) to 1 (free).
        #    task_value  = cost/ceiling      : the agent's own cost fraction is the
        #                  task's "expectation bar" for this resource. A resource
        #                  that is cheaper relative to the ceiling sits agent > task
        #                  (acceptance) while one that is expensive sits agent <= task
        #                  (rejection). This makes the formula sensitive to *how cheap*
        #                  the resource is, not just whether it fits under the ceiling.
        ceiling = dna.constraints.cost_ceiling_usd
        cost = manifest.cost_model.estimate_usd
        if ceiling > 0.0:
            cost_fraction = min(cost / ceiling, 1.0)        # 0 = free, 1 = at ceiling
            agent_val_cost = 1.0 - cost_fraction            # agent headroom: 1=free, 0=at ceiling
            task_val_cost = cost_fraction                   # task bar: how expensive this resource is
        else:
            # Ceiling is zero: only free resources are acceptable.
            agent_val_cost = 1.0 if cost == 0.0 else 0.0
            task_val_cost = 0.0 if cost == 0.0 else 1.0
        _record("cost", agent_val_cost, task_val_cost, w.cost)

        # 4. Latency — faster resource = higher agent_value (same encoding as cost).
        #
        #    agent_value = 1 - p95/slo : 1 = instant, 0 = exactly at SLO.
        #    task_value  = p95/slo     : latency fraction; fast resource has agent > task.
        slo = dna.constraints.latency_slo_ms
        p95 = manifest.latency_model.p95_ms
        if slo > 0:
            lat_fraction = min(p95 / slo, 1.0)
            agent_val_lat = 1.0 - lat_fraction
            task_val_lat = lat_fraction
        else:
            agent_val_lat = 1.0 if p95 == 0 else 0.0
            task_val_lat = 0.0 if p95 == 0 else 1.0
        _record("latency", agent_val_lat, task_val_lat, w.latency)

        # 5. Quality — agent's mean quality prior vs task's effective minimum.
        #
        #    agent_value = quality_for(flags)      actual prior for these flags
        #    task_value  = effective_min_quality() minimum the task requires
        #    A strong resource has agent > task (acceptance); a weak one has
        #    agent <= task (rejection).
        agent_val_q = manifest.quality_for(dna.flags)
        task_val_q = dna.effective_min_quality()
        _record("quality", agent_val_q, task_val_q, w.quality)

        score = (
            acceptance_rate * self.pessimising_factor
            - rejection_rate * self.optimising_factor
        )

        return DNAScore(
            resource_id=manifest.resource_id,
            score=score,
            acceptance_rate=acceptance_rate,
            rejection_rate=rejection_rate,
            dimensions=dimensions,
            quality=manifest.quality_for(dna.flags),
            cost_usd=cost,
            latency_ms=p95,
        )

    # -- scheduling: select + bind ------------------------------------------

    def select(self, dna: CapabilityDNA) -> SelectionResult:
        """Score every available resource against the DNA and return the winner.

        Availability (status="up") is the only hard gate: a resource that is
        down simply cannot serve the task. All other capability gaps are
        expressed as rejection-rate contributions to the score.

        Raises InfeasibleDNAError only when the registry is empty or every
        registered resource is unavailable.
        """
        if not self._manifests:
            raise InfeasibleDNAError(dna, {})

        # Pre-pass: filter out resources that are simply not running.
        available: List[CapabilityManifest] = []
        unavailable: Dict[str, str] = {}
        for rid, manifest in self._manifests.items():
            if manifest.availability.status == "up":
                available.append(manifest)
            else:
                unavailable[rid] = (
                    f"unavailable (status={manifest.availability.status})"
                )

        if not available:
            raise InfeasibleDNAError(dna, unavailable)

        # Score every available resource.
        scores: List[DNAScore] = [
            self.score_against_dna(manifest, dna) for manifest in available
        ]

        # Sort descending by score, tie-break on resource_id for determinism.
        scores.sort(key=lambda s: (-s.score, s.resource_id))

        winner = scores[0]
        return SelectionResult(
            resource_id=winner.resource_id,
            score=winner.score,
            quality=winner.quality,
            cost_usd=winner.cost_usd,
            latency_ms=winner.latency_ms,
            all_scores=scores,
        )

    def bind(self, dna: CapabilityDNA) -> Tuple[SelectionResult, Callable]:
        """select() plus the callable, which is what the executor actually needs."""
        decision = self.select(dna)
        return decision, self._run_fns[decision.resource_id]

    # -- legacy M5 methods kept for compatibility ---------------------------

    def feasible(
        self, dna: CapabilityDNA
    ) -> Tuple[List[CapabilityManifest], Dict[str, str]]:
        """Legacy feasibility filter shim — kept so old test code compiles.

        In the new design there is no hard feasibility filter beyond the
        availability gate. This shim returns every available resource as a
        'survivor' and the unavailable ones as rejections, which is the
        closest approximation to the old contract without reintroducing the
        hard gates.
        """
        survivors: List[CapabilityManifest] = []
        rejections: Dict[str, str] = {}
        for rid, manifest in self._manifests.items():
            if manifest.availability.status == "up":
                survivors.append(manifest)
            else:
                rejections[rid] = (
                    f"unavailable (status={manifest.availability.status})"
                )
        return survivors, rejections

    def score_candidate(
        self, manifest: CapabilityManifest, dna: CapabilityDNA
    ) -> DNAScore:
        """Legacy shim — delegates to the new continuous scorer."""
        return self.score_against_dna(manifest, dna)

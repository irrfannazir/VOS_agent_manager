"""Capability Registry: resource manifests + the two-stage scheduling decision.

DOC1 5.1 defines what a resource publishes (CapabilityManifest). DOC1 M5
defines how a subtask's Capability DNA is turned into a concrete resource
binding, in two stages:

  1. Feasibility filter -- resource.capabilities is a superset of dna.flags,
     AND the cost / latency / quality / risk constraints all hold.
  2. Policy scoring -- Pareto scorer over the survivors:
     score = quality_prior - lambda*cost_est - mu*latency_est

Stage 2 is deliberately rule-based here; the contextual bandit that replaces it
plugs in at `score_candidate` without touching stage 1.
"""

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


class InfeasibleDNAError(RuntimeError):
    """No registered resource satisfies a subtask's DNA.

    Carries the structured reason per resource so admission control (and the
    paper's rejection analysis) can report *which* constraint bit, rather than
    just "no match".
    """

    def __init__(self, dna: CapabilityDNA, rejections: Dict[str, str]):
        self.dna = dna
        self.rejections = rejections
        detail = "; ".join(f"{name}: {why}" for name, why in sorted(rejections.items()))
        super().__init__(
            f"[capability-registry] no feasible resource for DNA "
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
# Scheduling decision records
# ---------------------------------------------------------------------------


class Candidate(BaseModel):
    """A resource that survived the feasibility filter, with its scored terms."""

    resource_id: str
    score: float
    quality: float
    cost_usd: float
    latency_ms: int
    # Normalised terms, kept so the trace shows *why* one beat another.
    cost_term: float
    latency_term: float


class ScheduleDecision(BaseModel):
    """Full audit record of one M5 decision. Every field is a span attribute."""

    resource_id: str
    score: float
    quality: float
    cost_usd: float
    latency_ms: int
    candidates: List[Candidate]
    rejections: Dict[str, str]
    runner_up: Optional[str] = None
    runner_up_margin: Optional[float] = None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class CapabilityRegistry:
    """Live manifest table + capability lookup + M5 two-stage binding.

    lambda_cost / mu_latency are the Pareto scorer's tradeoff weights. Both
    terms are normalised against the subtask's own ceiling and SLO before
    weighting, so the three terms in `score` are commensurable regardless of
    whether a task is budget-bound or latency-bound.
    """

    def __init__(self, lambda_cost: float = 0.3, mu_latency: float = 0.2) -> None:
        self._manifests: Dict[str, CapabilityManifest] = {}
        self._run_fns: Dict[str, Callable] = {}
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

    # -- M5 stage 1: feasibility filter -------------------------------------

    def feasible(
        self, dna: CapabilityDNA
    ) -> Tuple[List[CapabilityManifest], Dict[str, str]]:
        """Split registered resources into (survivors, {resource_id: reason}).

        Checks run cheapest-first and short-circuit, so the recorded reason is
        the *first* constraint violated rather than an exhaustive list. That is
        what makes the rejection log readable.
        """
        survivors: List[CapabilityManifest] = []
        rejections: Dict[str, str] = {}
        min_quality = dna.effective_min_quality()
        tolerance = risk_rank(dna.constraints.risk_tolerance)

        for rid, manifest in self._manifests.items():
            if manifest.availability.status != "up":
                rejections[rid] = f"unavailable (status={manifest.availability.status})"
                continue

            missing = missing_flags(dna, manifest.capabilities)
            if missing:
                rejections[rid] = f"missing capability {missing}"
                continue

            if risk_rank(manifest.risk_class) > tolerance:
                rejections[rid] = (
                    f"risk_class={manifest.risk_class} exceeds tolerance="
                    f"{dna.constraints.risk_tolerance}"
                )
                continue

            cost = manifest.cost_model.estimate_usd
            if cost > dna.constraints.cost_ceiling_usd:
                rejections[rid] = (
                    f"cost ${cost:.4f} over ceiling "
                    f"${dna.constraints.cost_ceiling_usd:.4f}"
                )
                continue

            latency = manifest.latency_model.p95_ms
            if latency > dna.constraints.latency_slo_ms:
                rejections[rid] = (
                    f"p95 {latency}ms over SLO {dna.constraints.latency_slo_ms}ms"
                )
                continue

            quality = manifest.quality_for(dna.flags)
            if quality < min_quality:
                rejections[rid] = (
                    f"quality {quality:.2f} below required {min_quality:.2f}"
                )
                continue

            survivors.append(manifest)

        return survivors, rejections

    # -- M5 stage 2: Pareto scoring -----------------------------------------

    def score_candidate(
        self, manifest: CapabilityManifest, dna: CapabilityDNA
    ) -> Candidate:
        """score = quality - lambda*cost_norm - mu*latency_norm.

        Cost and latency are normalised by the subtask's own ceiling and SLO,
        putting both in [0,1] for any resource that passed stage 1. A zero
        ceiling or SLO would divide by zero; treat those as "no headroom" and
        pin the term to 0, since stage 1 already proved the resource fits.
        """
        quality = manifest.quality_for(dna.flags)
        cost = manifest.cost_model.estimate_usd
        latency = manifest.latency_model.p95_ms

        ceiling = dna.constraints.cost_ceiling_usd
        slo = dna.constraints.latency_slo_ms
        cost_term = (cost / ceiling) if ceiling > 0 else 0.0
        latency_term = (latency / slo) if slo > 0 else 0.0

        score = quality - self.lambda_cost * cost_term - self.mu_latency * latency_term

        return Candidate(
            resource_id=manifest.resource_id,
            score=score,
            quality=quality,
            cost_usd=cost,
            latency_ms=latency,
            cost_term=cost_term,
            latency_term=latency_term,
        )

    def select(self, dna: CapabilityDNA) -> ScheduleDecision:
        """Run both stages and return the winning binding with its audit trail."""
        if not self._manifests:
            raise InfeasibleDNAError(dna, {})

        survivors, rejections = self.feasible(dna)
        if not survivors:
            raise InfeasibleDNAError(dna, rejections)

        candidates = sorted(
            (self.score_candidate(m, dna) for m in survivors),
            # Tie-break on resource_id so identical scores bind deterministically
            # -- required for the seeded-replay determinism audit.
            key=lambda c: (-c.score, c.resource_id),
        )

        winner = candidates[0]
        runner_up = candidates[1] if len(candidates) > 1 else None

        return ScheduleDecision(
            resource_id=winner.resource_id,
            score=winner.score,
            quality=winner.quality,
            cost_usd=winner.cost_usd,
            latency_ms=winner.latency_ms,
            candidates=candidates,
            rejections=rejections,
            runner_up=runner_up.resource_id if runner_up else None,
            runner_up_margin=(winner.score - runner_up.score) if runner_up else None,
        )

    def bind(self, dna: CapabilityDNA) -> Tuple[ScheduleDecision, Callable]:
        """select() plus the callable, which is what the executor actually needs."""
        decision = self.select(dna)
        return decision, self._run_fns[decision.resource_id]

    # -- admission control seed (DOC1 M2) -----------------------------------

    def unsatisfiable_flags(self, dna: CapabilityDNA) -> List[str]:
        """Flags no registered resource provides at all.

        Distinct from feasibility: this is the hard, pre-execution check the
        Task Manager uses to reject a plan by *naming the missing capability*,
        rather than a soft constraint miss that a cheaper resource might clear.
        """
        return missing_flags(dna, self.provided_flags())

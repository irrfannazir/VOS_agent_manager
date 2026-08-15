"""Core data models.

The Capability DNA here follows DOC1 section 5.2: a *per-subtask requirement*
vector, not a per-resource fit score. Resources describe themselves with a
CapabilityManifest (DOC1 5.1) which lives in capability_registry.py.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Capability flag vocabulary
# ---------------------------------------------------------------------------
# Discrete capability identifiers. Dotted "family.specific" naming, per DOC1
# 5.2. A subtask's DNA lists the flags it requires; a resource's manifest lists
# the flags it provides. The feasibility filter is set containment over these,
# so both sides MUST draw from this one vocabulary or nothing ever matches.
CAPABILITY_FLAGS = [
    "reasoning.deep",
    "reasoning.shallow",
    "planning.decomposition",
    "vision.understanding",
    "speech.transcription",
    "speech.synthesis",
    "code.generation",
    "tool.calling",
    "web.search",
    "db.query",
    "sensor.reading",
    "text.summarization",
    "text.classification",
    # Writing the user-facing final answer at full detail. Distinct from
    # text.summarization on purpose: that flag selects for compression, and the
    # terminal node needs the opposite. Without its own flag the terminal node
    # scores identically to a summarize node and binds to the summarizer.
    "answer.synthesis",
    "domain.specific",
]

# Ordinal axes, scored 0-4. These describe *how hard* the subtask is along each
# axis, independent of which capabilities it needs.
ORDINAL_FIELDS = [
    "reasoning_depth",
    "planning_horizon",
    "tool_complexity",
    "memory_dependence",
    "parallelizability",
]

# Ordered risk ladder. A resource is admissible only when its risk_class sits at
# or below the subtask's risk_tolerance, so the ordering is load-bearing.
RISK_LEVELS = ["low", "medium", "high"]

RiskLevel = Literal["low", "medium", "high"]


def risk_rank(level: str) -> int:
    """Position on the risk ladder. Unknown levels sort as the most dangerous."""
    try:
        return RISK_LEVELS.index(level)
    except ValueError:
        return len(RISK_LEVELS)


# ---------------------------------------------------------------------------
# Capability DNA (DOC1 5.2)
# ---------------------------------------------------------------------------


class DNAOrdinals(BaseModel):
    """Ordinal complexity scores, each 0-4."""

    reasoning_depth: int = Field(default=0, ge=0, le=4)
    planning_horizon: int = Field(default=0, ge=0, le=4)
    tool_complexity: int = Field(default=0, ge=0, le=4)
    memory_dependence: int = Field(default=0, ge=0, le=4)
    parallelizability: int = Field(default=0, ge=0, le=4)

    def demand(self) -> float:
        """Normalised 0-1 difficulty over the axes that imply *capability need*.

        parallelizability is a scheduling hint, not a difficulty signal, and
        memory_dependence is about state plumbing rather than raw model
        strength — so neither feeds the demand score. The remaining three axes
        max out at 4 each, hence the /12.
        """
        return (
            self.reasoning_depth + self.planning_horizon + self.tool_complexity
        ) / 12.0


class DNAConstraints(BaseModel):
    """Continuous execution constraints for a single subtask.

    Defaults are deliberately permissive: an un-extracted node should execute,
    not get rejected by admission control before the extractor has run.
    """

    cost_ceiling_usd: float = Field(default=1.0, ge=0.0)
    latency_slo_ms: int = Field(default=120_000, ge=0)
    min_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    risk_tolerance: RiskLevel = "medium"


class CapabilityDNA(BaseModel):
    """Typed per-subtask requirement vector (DOC1 5.2).

    Three segments: discrete capability flags, ordinal complexity scores, and
    continuous constraints. `confidence` and `extracted_by` are provenance
    fields the DNA extractor fills in — they drive the confidence-gated
    escalation to a stronger model and end up as span attributes.
    """

    flags: List[str] = Field(default_factory=list)
    ordinals: DNAOrdinals = Field(default_factory=DNAOrdinals)
    constraints: DNAConstraints = Field(default_factory=DNAConstraints)

    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    extracted_by: Optional[str] = None

    @field_validator("flags")
    @classmethod
    def _flags_in_vocabulary(cls, flags: List[str]) -> List[str]:
        unknown = [f for f in flags if f not in CAPABILITY_FLAGS]
        if unknown:
            raise ValueError(
                f"unknown capability flag(s) {unknown}; "
                f"vocabulary is {CAPABILITY_FLAGS}"
            )
        # De-duplicate while preserving order — the filter is set-based, but a
        # stable list keeps plan.md and trace attributes readable.
        seen: set[str] = set()
        return [f for f in flags if not (f in seen or seen.add(f))]

    def effective_min_quality(self) -> float:
        """Quality bar a resource must clear, after folding in the ordinals.

        A subtask declaring reasoning_depth=4 is asking for a strong model even
        if the planner left min_quality at its default. Taking the max of the
        declared floor and the ordinal demand is what gives the ordinal segment
        real teeth in the feasibility filter.
        """
        return max(self.constraints.min_quality, self.ordinals.demand())


# ---------------------------------------------------------------------------
# Task DAG (DOC1 5.3)
# ---------------------------------------------------------------------------


class Node(BaseModel):
    id: str
    description: str
    # Coarse capability string kept from v0.0.2. It is the exact-match fallback
    # used when a node has no DNA (extractor failure, or DNA routing disabled).
    capability: str
    depends_on: List[str] = []
    input: Optional[str] = None
    output: Optional[str] = None
    status: str = "pending"  # pending | running | done | failed
    performed_by: Optional[str] = None

    # Filled by the DNA extractor (M4), consumed by the scheduler (M5).
    dna: Optional[CapabilityDNA] = None
    # Which resource the scheduler actually bound, and why. Distinct from
    # performed_by, which names the sub-agent rather than the resource.
    bound_resource: Optional[str] = None
    routing_mode: Optional[str] = None  # "dna" | "exact"


class Graph(BaseModel):
    job: str
    nodes: List[Node]

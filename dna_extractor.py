"""M4 -- Capability DNA Extractor.

A dedicated extraction pass, separate from planning: each subtask description
is mapped to the *judgement* segments of a Capability DNA vector (DOC1 5.2) by
schema-constrained generation. Per DOC1 M4 the extractor is itself a routed
resource -- a cheap model runs first and only low-confidence subtasks are
escalated to a strong model. That escalation is the whole reason `confidence`
lives on the DNA.

Scope note: this extractor emits flags + ordinals + confidence ONLY. The
constraint segment is derived by constraint_policy.ConstraintPolicy from the
job budget and the live resource pool. The first live run is why: asked to
invent budgets, the cheap model returned `latency_slo_ms: 500` for every node,
which is below the p95 of every registered resource, so the feasibility filter
rejected everything and DNA routing never fired once.

Extraction never hard-fails the pipeline: if both models are unavailable or
return garbage, a keyword heuristic supplies a conservative DNA and marks its
provenance, so the node still executes (via the exact-match fallback if the
heuristic produced nothing useful).
"""

import json
from typing import List, Optional

from groq import Groq

from config import GROQ_API_KEY
from models import (
    CAPABILITY_FLAGS,
    ORDINAL_FIELDS,
    CapabilityDNA,
    DNAConstraints,
    DNAOrdinals,
    Graph,
    Node,
)

# Cheap model handles the common case; the strong model is the escalation arm.
_CHEAP_MODEL = "llama-3.1-8b-instant"
_STRONG_MODEL = "llama-3.3-70b-versatile"

_FLAGS_LIST = ", ".join(f'"{f}"' for f in CAPABILITY_FLAGS)

_SYSTEM_PROMPT = f"""\
You extract a Capability DNA vector for ONE subtask of a larger job.

You produce TWO segments:

1. flags -- the capability identifiers this subtask genuinely REQUIRES, chosen \
only from this fixed vocabulary: {_FLAGS_LIST}
   Pick 1-2 (rarely 3). Do not check every box. A resource is eligible only if \
it provides EVERY flag you list, so an over-broad list makes the subtask \
unschedulable. Pick the narrowest set that is actually required.

   Guidance:
   - fetching facts from the internet -> ["web.search"]
   - condensing text you were handed -> ["text.summarization"]
   - comparing, contrasting, judging, or drawing conclusions across sources \
-> ["reasoning.deep"] (add "text.summarization" only if the output must also \
be condensed)
   - writing the final answer to the user's whole job -> \
["reasoning.deep", "text.summarization"]
   - describing an image -> ["vision.understanding"]

2. ordinals -- integers 0-4 scoring how demanding the subtask is:
   - reasoning_depth: 0 = lookup/copy, 4 = multi-step novel inference
   - planning_horizon: 0 = single shot, 4 = long multi-stage plan
   - tool_complexity: 0 = no tools, 4 = many chained tool calls
   - memory_dependence: 0 = self-contained, 4 = needs lots of prior context
   - parallelizability: 0 = strictly sequential, 4 = trivially parallel

Do NOT emit budgets, deadlines, costs or quality thresholds. You have no way \
to know the system's spend envelope or the latency of its resources; the \
kernel derives all of that from the ordinals you produce.

Also report `confidence`: 0.0-1.0, how sure you are of THIS extraction. Be \
honest -- low confidence triggers a second opinion from a stronger model, \
which is cheap; a confidently wrong vector is not.

You MUST call the extract_dna function. Do not respond with plain text."""

_EXTRACT_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_dna",
        "description": "Submit the Capability DNA for the given subtask.",
        "parameters": {
            "type": "object",
            "properties": {
                "flags": {
                    "type": "array",
                    "items": {"type": "string", "enum": CAPABILITY_FLAGS},
                    "description": "1-3 required capability flags.",
                },
                "ordinals": {
                    "type": "object",
                    "properties": {
                        name: {"type": "integer", "minimum": 0, "maximum": 4}
                        for name in ORDINAL_FIELDS
                    },
                    "required": ORDINAL_FIELDS,
                },
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
            "required": ["flags", "ordinals", "confidence"],
        },
    },
}

# Keyword -> flag table for the offline fallback. Ordered longest-intent first
# so "summar" wins over a bare "search" when a description mentions both.
_HEURISTIC_KEYWORDS: list[tuple[tuple[str, ...], str]] = [
    (("image", "photo", "picture", "visual", "diagram"), "vision.understanding"),
    (("transcri", "audio", "speech-to-text"), "speech.transcription"),
    (("code", "function", "script", "implement", "refactor"), "code.generation"),
    (("summar", "condense", "digest", "tl;dr"), "text.summarization"),
    (("search", "look up", "research", "find information"), "web.search"),
    (("classif", "categor", "label", "tag"), "text.classification"),
    (("query", "database", "sql", "table"), "db.query"),
    (("compare", "analyz", "analys", "reason", "explain", "evaluate"), "reasoning.deep"),
    (("synthes", "final answer", "combine", "assemble"), "reasoning.deep"),
]

# Coarse capability -> flag, used to seed the heuristic when the node's
# description yields no keyword hit at all.
_CAPABILITY_TO_FLAG = {
    "web_search": "web.search",
    "summarization": "text.summarization",
    "vision": "vision.understanding",
    "synthesis": "reasoning.deep",
}


class DNAExtractor:
    """Maps subtask descriptions to Capability DNA (DOC1 M4).

    confidence_threshold is the escalation gate: anything the cheap model is
    less sure of than this gets re-extracted by the strong model, and the more
    confident of the two vectors wins.
    """

    def __init__(
        self,
        cheap_model: str = _CHEAP_MODEL,
        strong_model: str = _STRONG_MODEL,
        confidence_threshold: float = 0.7,
        client: Optional[Groq] = None,
    ) -> None:
        self.cheap_model = cheap_model
        self.strong_model = strong_model
        self.confidence_threshold = confidence_threshold
        self._client = client or Groq(api_key=GROQ_API_KEY)

    # -- public API ---------------------------------------------------------

    def extract_graph(self, graph: Graph) -> Graph:
        """Fill `dna` on every node in place, then return the graph."""
        print(
            f"[dna-extractor] extracting Capability DNA for "
            f"{len(graph.nodes)} node(s)"
        )
        escalated = 0
        for node in graph.nodes:
            # Kernel-authored nodes (the terminal synthesis node) arrive with
            # their DNA already set. Re-extracting would let the model overwrite
            # a binding the kernel guarantees.
            if node.dna is not None and node.dna.extracted_by == "kernel":
                print(f"[dna-extractor] node '{node.id}': kernel-authored DNA, skipping")
                continue
            node.dna = self.extract(node, graph.job)
            if node.dna.extracted_by == self.strong_model:
                escalated += 1
        print(
            f"[dna-extractor] done ({escalated}/{len(graph.nodes)} escalated to "
            f"'{self.strong_model}')"
        )
        return graph

    def extract(self, node: Node, job: str) -> CapabilityDNA:
        """Extract one node's DNA, escalating on low confidence."""
        dna = self._try_model(self.cheap_model, node, job)

        if dna is not None and dna.confidence >= self.confidence_threshold:
            self._log(node, dna, escalated=False)
            return dna

        reason = (
            "extraction failed"
            if dna is None
            else f"confidence {dna.confidence:.2f} < {self.confidence_threshold:.2f}"
        )
        print(f"[dna-extractor] node '{node.id}': {reason} -> escalating")

        strong = self._try_model(self.strong_model, node, job)

        # Keep whichever attempt was more confident; the strong model is the
        # tie-breaker when neither is clearly better.
        best = strong if strong is not None else dna
        if strong is not None and dna is not None and dna.confidence > strong.confidence:
            best = dna

        if best is None:
            best = self._heuristic_dna(node)
            print(
                f"[dna-extractor] node '{node.id}': both models failed, "
                f"using heuristic DNA"
            )

        self._log(node, best, escalated=True)
        return best

    # -- internals ----------------------------------------------------------

    def _try_model(self, model: str, node: Node, job: str) -> Optional[CapabilityDNA]:
        """One constrained extraction attempt. Returns None on any failure.

        Swallowing exceptions here is intentional: a dead model or a malformed
        tool call must degrade to escalation or heuristic, never kill the run.
        """
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"OVERALL JOB: {job}\n\n"
                            f"SUBTASK id={node.id}\n"
                            f"SUBTASK DESCRIPTION: {node.description}\n"
                            f"COARSE CAPABILITY HINT: {node.capability}\n"
                            f"DEPENDS ON {len(node.depends_on)} upstream node(s)."
                        ),
                    },
                ],
                tools=[_EXTRACT_TOOL],
                tool_choice={"type": "function", "function": {"name": "extract_dna"}},
                temperature=0.0,
            )
            payload = self._extract_fn_call(response)
            return self._build_dna(payload, extracted_by=model)
        except Exception as exc:  # noqa: BLE001 -- degradation is the contract
            print(f"[dna-extractor] node '{node.id}': model '{model}' failed: {exc}")
            return None

    @staticmethod
    def _extract_fn_call(response) -> dict:
        message = response.choices[0].message
        if not message.tool_calls:
            raise RuntimeError(
                f"model returned no extract_dna call. Content: {message.content}"
            )
        return json.loads(message.tool_calls[0].function.arguments)

    @staticmethod
    def _build_dna(payload: dict, extracted_by: str) -> CapabilityDNA:
        """Validate a raw tool-call payload into a CapabilityDNA.

        Unknown flags are dropped rather than raising: models occasionally
        invent a plausible-looking flag, and losing one flag is recoverable
        while losing the whole extraction is not.

        Any `constraints` key in the payload is ignored. The schema does not
        ask for one, but a model that hallucinates the field must not be able
        to smuggle a 500 ms SLO past the kernel -- constraint_policy owns that
        segment outright.
        """
        raw_flags = payload.get("flags") or []
        flags = [f for f in raw_flags if f in CAPABILITY_FLAGS]
        dropped = [f for f in raw_flags if f not in CAPABILITY_FLAGS]
        if dropped:
            print(f"[dna-extractor] dropped out-of-vocabulary flag(s) {dropped}")

        raw_ordinals = payload.get("ordinals") or {}
        ordinals = DNAOrdinals(
            **{k: v for k, v in raw_ordinals.items() if k in ORDINAL_FIELDS}
        )

        return CapabilityDNA(
            flags=flags,
            ordinals=ordinals,
            constraints=DNAConstraints(),  # placeholder; ConstraintPolicy fills it
            confidence=float(payload.get("confidence", 0.0)),
            extracted_by=extracted_by,
        )

    @staticmethod
    def _heuristic_dna(node: Node) -> CapabilityDNA:
        """Keyword fallback. Deliberately permissive on constraints.

        This path exists to keep the pipeline running when the extractor models
        are unreachable, so it must not manufacture constraints tight enough to
        make the node infeasible. Confidence is pinned low to mark the DNA as
        untrustworthy in traces and in the agreement study.
        """
        text = f"{node.description} {node.capability}".lower()

        flags: List[str] = []
        for keywords, flag in _HEURISTIC_KEYWORDS:
            if any(k in text for k in keywords) and flag not in flags:
                flags.append(flag)

        if not flags:
            seeded = _CAPABILITY_TO_FLAG.get(node.capability)
            if seeded:
                flags = [seeded]

        return CapabilityDNA(
            flags=flags[:2],
            ordinals=DNAOrdinals(
                reasoning_depth=2 if "reasoning.deep" in flags else 1,
                memory_dependence=1 if node.depends_on else 0,
                parallelizability=0 if node.depends_on else 3,
            ),
            constraints=DNAConstraints(),
            confidence=0.2,
            extracted_by="heuristic",
        )

    @staticmethod
    def _log(node: Node, dna: CapabilityDNA, escalated: bool) -> None:
        tag = "escalated" if escalated else "cheap"
        # Constraints are still placeholders at this point, so log only what the
        # extractor actually decided.
        print(
            f"[dna-extractor] node '{node.id}' [{tag}/{dna.extracted_by}] "
            f"flags={dna.flags} demand={dna.ordinals.demand():.2f} "
            f"conf={dna.confidence:.2f}"
        )

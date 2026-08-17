import json
from pathlib import Path
from typing import Optional

from groq import Groq

from aos_v0.config import GROQ_API_KEY
from aos_v0.graph_utils import (
    validate_graph,
    build_waves,
    get_sink_nodes,
    GraphValidationError,
)
from aos_v0.diagram_utils import build_mermaid
from aos_v0.models import CapabilityDNA, DNAOrdinals, Graph, Node

# The planner may only choose among these. "synthesis" is deliberately absent:
# the kernel appends the terminal synthesis node itself.
_CAPABILITIES = ["web_search", "summarization", "vision", "speech_transcription", "audio"]

_SYNTHESIS_NODE_ID = "final"

_MODEL = "openai/gpt-oss-120b"

_SYSTEM_PROMPT = f"""\
You are a task planner. Given a user job, decompose it into a directed acyclic \
graph (DAG) of tasks.

Each node must have:
- id: a short lowercase letter or string identifier (e.g. "a", "b", "search1")
- description: a SPECIFIC, ACTIONABLE instruction for the node's capability. \
Not vague — this text IS the instruction passed to the capability. E.g. \
"Search for recent advances in solar panel efficiency" not "do research".
- capability: EXACTLY ONE from this fixed list: {_CAPABILITIES}
- depends_on: a list of node ids this node depends on. Empty list [] means \
root node (no dependencies).

Do NOT reason about which model, tool or resource should run a node, and do \
not emit capability requirements or cost/latency budgets. A separate Capability \
DNA extraction pass handles that. Your only job is the decomposition.

DO NOT create a final "combine", "synthesise" or "write the answer" node. The \
kernel appends that node itself, wired to every leaf of your graph. Your last \
nodes should be the ones that produce the raw material for it.

GATHER AT FULL DETAIL: when the job asks for a specific quantity ("5 news \
items") or for completeness ("all match results"), the node description MUST \
repeat that requirement verbatim, because the description becomes the search \
query and the instruction. Write "Find 5 current news stories about the FIFA \
president", never "Research the FIFA president".

PARALLELISM: When two or more nodes do not depend on each other, they can run \
in parallel. You SHOULD use multiple parallel branches when the task naturally \
splits (e.g. researching two different topics at the same time).

MERGING: A node can depend on MULTIPLE parents. Its input will be the \
concatenated outputs of all parent nodes. Use this for nodes that need to \
combine or compare results from parallel branches.

CRITICAL — ONE NODE PER NAMED ENTITY: When the job names multiple distinct \
entities that each need individual research, analysis, or comparison (e.g. a \
list of teams, products, companies, people, places), create ONE web_search node \
AND ONE summarization node PER NAMED ENTITY — never combine multiple named \
entities into a single node. Only merge individual entity results together at a \
later, separate comparison/aggregation node that depends on all of them.

Example: if the job is "analyze Brazil, Portugal, and Spain", produce:
  - 3 web_search nodes (one for Brazil, one for Portugal, one for Spain)
  - 3 summarization nodes (one per team, each depending on its own web_search)
  - optionally 1 comparison node depending on all 3 summarization nodes
NOT a single node covering all three teams.

If an image, audio, or other media file is provided in the job, include root \
node(s) using the appropriate capability that process the media, feeding into \
downstream nodes. When the job asks for MULTIPLE independent analyses of the \
same media (e.g. "transcribe AND identify audio events"), create SEPARATE \
parallel root nodes — one per distinct analysis task — all depending on [] \
(all are root nodes). They will receive the original media artifact. Do NOT \
merge independent analyses into a single node. If no media is mentioned, do \
NOT include a media-processing node.

The id for each node must be a simple string like "a", "b", "c", etc.

You MUST call the create_graph function with the decomposed graph. Do not \
respond with plain text."""

_CREATE_GRAPH_TOOL = {
    "type": "function",
    "function": {
        "name": "create_graph",
        "description": "Submit the decomposed task graph.",
        "parameters": {
            "type": "object",
            "properties": {
                "job": {
                    "type": "string",
                    "description": "The original user job.",
                },
                "nodes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "description": {"type": "string"},
                            "capability": {"type": "string"},
                            "depends_on": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["id", "description", "capability", "depends_on"],
                    },
                },
            },
            "required": ["job", "nodes"],
        },
    },
}

_README_SECTION = """\
## How AOS v0.0.3 plans and executes tasks

AOS decomposes a user job into a **task graph** — a directed acyclic graph
(DAG) of **nodes** connected by **edges**.

- **Node:** a single unit of work. Each node has a unique `id`, a `description`
  (which doubles as the instruction passed to its capability), a `capability`
  (the coarse tool hint, e.g. `web_search`, `summarization`, or `vision`), a
  **Capability DNA** vector, and a `depends_on` list of node ids it must wait
  for.
- **Edge:** a dependency from one node to another. If node C has
  `depends_on: ["a", "b"]`, it runs only after both A and B finish, and
  receives their outputs as input.

**Capability DNA (DOC1 §5.2):** planning and resource selection are separate
passes. The Manager Agent only decomposes; a dedicated **DNA Extractor** then
maps each subtask to a typed requirement vector with three segments —
discrete capability `flags`, `ordinals` scoring difficulty 0–4, and
`constraints` (cost ceiling, latency SLO, min quality, risk tolerance). The
extractor runs a cheap model first and escalates to a strong one only when its
self-reported confidence falls below threshold.

**Two-stage binding (DOC1 M5):** for each node the Capability Registry first
applies a **feasibility filter** — a resource must provide every flag in the
DNA and satisfy all four constraints — then scores the survivors with a Pareto
scorer, `score = quality_prior − λ·cost − μ·latency`, with cost and latency
normalised against that subtask's own ceiling and SLO. The winner, the
runner-up margin, and the reason each rejected resource was filtered are all
logged.

**Concurrency:** nodes in the same dependency "wave" are independent and run
concurrently via threads. For example, in the graph below, nodes `a` and `b`
run in parallel, then `c` and `d` run in parallel, and finally `e` runs.

**Multi-parent merging:** a node with multiple parents receives the labeled,
concatenated outputs of all its parents, so it can compare, combine, or
reconcile them.

**Diagrams:** the Mermaid diagrams in `outputs/plan.md` are generated
deterministically from the validated graph by `diagram_utils.build_mermaid()`,
not by the LLM. This is a deliberate reliability choice — the diagram always
faithfully represents the actual graph structure, wave assignments, and
capability types, with no risk of the LLM inventing or omitting nodes.

```mermaid
flowchart TD
    subgraph Wave 0
        a["Research solar energy (web_search)"]
        b["Research wind energy (web_search)"]
    end
    subgraph Wave 1
        c["Summarize solar (summarization)"]
        d["Summarize wind (summarization)"]
    end
    subgraph Wave 2
        e["Compare summaries (summarization)"]
    end
    a --> c
    b --> d
    c --> e
    d --> e

    classDef web_searchStyle fill:#4CAF50,color:#fff
    classDef summarizationStyle fill:#2196F3,color:#fff
    class a web_searchStyle
    class b web_searchStyle
    class c summarizationStyle
    class d summarizationStyle
    class e summarizationStyle
```
"""


class ManagerAgent:
    _MAX_STRUCTURAL_RETRIES = 1
    _MAX_COMPLETENESS_RETRIES = 1

    def __init__(self):
        self._client = Groq(api_key=GROQ_API_KEY)

    def create_plan(self, user_prompt: str, inputs: Optional[dict[str, str]] = None) -> Graph:
        print("[manager-agent] decomposing job into a task graph...")

        graph = self._call_llm(user_prompt, inputs)

        # --- structural validation loop ---
        structural_attempts = 0
        while True:
            try:
                validate_graph(graph)
                break
            except GraphValidationError as exc:
                structural_attempts += 1
                if structural_attempts > self._MAX_STRUCTURAL_RETRIES:
                    raise RuntimeError(
                        f"Graph still invalid after {self._MAX_STRUCTURAL_RETRIES} "
                        f"structural retry: {exc}"
                    ) from exc
                print(f"[manager-agent] structural validation failed, retrying ({structural_attempts}/{self._MAX_STRUCTURAL_RETRIES}): {exc}")
                graph = self._call_llm_with_retry(user_prompt, inputs, str(exc))

        # --- semantic completeness check ---
        completeness_attempts = 0
        while True:
            verdict = self._check_completeness(user_prompt, graph)
            if verdict == "COMPLETE":
                print("[manager-agent] completeness check: COMPLETE")
                break
            completeness_attempts += 1
            if completeness_attempts > self._MAX_COMPLETENESS_RETRIES:
                raise RuntimeError(
                    f"Graph still incomplete after {self._MAX_COMPLETENESS_RETRIES} "
                    f"completeness retry: {verdict}"
                )
            print(f"[manager-agent] completeness check: INCOMPLETE ({verdict}) — regenerating")
            graph = self._call_llm_with_completeness_retry(
                user_prompt, inputs, verdict
            )
            # re-validate structurally after completeness regeneration
            try:
                validate_graph(graph)
            except GraphValidationError as exc:
                print(f"[manager-agent] structural validation failed after completeness retry: {exc}")
                graph = self._call_llm_with_retry(user_prompt, inputs, str(exc))
                validate_graph(graph)

        self._append_synthesis_node(graph, user_prompt)
        validate_graph(graph)

        self._write_plan_md(graph)
        self._ensure_readme()

        waves = build_waves(graph)
        print(
            "[manager-agent] graph written to outputs/plan.md "
            f"({len(graph.nodes)} nodes, {len(waves)} waves)"
        )
        return graph

    @staticmethod
    def _append_synthesis_node(graph: Graph, user_prompt: str) -> None:
        """Wire a terminal synthesis node to every leaf of the planned graph.

        Done in code rather than asked of the planner for two reasons. First,
        reliability: the final answer node is the one node whose absence ruins
        the run, so it should not depend on the LLM remembering to emit it.
        Second, fidelity: its description is set to the user's *verbatim*
        original prompt, which is what the synthesis capability answers against.
        A planner-written paraphrase ("Combine the summaries from nodes e, f, g
        and h") loses the user's actual asks -- the quantities, the entities,
        the questions -- which is exactly how the first run lost them.
        """
        existing = {n.id for n in graph.nodes}
        node_id = _SYNTHESIS_NODE_ID
        suffix = 2
        while node_id in existing:
            node_id = f"{_SYNTHESIS_NODE_ID}{suffix}"
            suffix += 1

        sinks = get_sink_nodes(graph)
        graph.nodes.append(
            Node(
                id=node_id,
                description=user_prompt,
                capability="synthesis",
                depends_on=[n.id for n in sinks],
                # Kernel-authored node, so the kernel authors its DNA too rather
                # than round-tripping through the extractor. answer.synthesis is
                # what makes this node bind to the synthesis resource; with only
                # reasoning.deep + text.summarization it scores the same as any
                # summarize node and binds to the summarizer instead.
                dna=CapabilityDNA(
                    flags=["answer.synthesis", "reasoning.deep"],
                    ordinals=DNAOrdinals(
                        reasoning_depth=4,
                        planning_horizon=2,
                        memory_dependence=4,
                        parallelizability=0,
                    ),
                    confidence=1.0,
                    extracted_by="kernel",
                ),
            )
        )
        print(
            f"[manager-agent] appended synthesis node '{node_id}' depending on "
            f"{[n.id for n in sinks]}"
        )

    def _call_llm(self, user_prompt: str, inputs: Optional[dict[str, str]] = None) -> Graph:
        messages = self._build_messages(user_prompt, inputs)
        response = self._client.chat.completions.create(
            model=_MODEL,
            messages=messages,
            tools=[_CREATE_GRAPH_TOOL],
            tool_choice={"type": "function", "function": {"name": "create_graph"}},
            temperature=0.2,
        )
        return Graph.model_validate(self._extract_fn_call(response))

    def _call_llm_with_retry(
        self, user_prompt: str, inputs: Optional[dict[str, str]], error_msg: str
    ) -> Graph:
        messages = self._build_messages(user_prompt, inputs)
        messages.append(
            {
                "role": "user",
                "content": (
                    "Your previous graph failed validation:\n"
                    f"{error_msg}\n\n"
                    "Please fix the errors and call create_graph again."
                ),
            }
        )
        response = self._client.chat.completions.create(
            model=_MODEL,
            messages=messages,
            tools=[_CREATE_GRAPH_TOOL],
            tool_choice={"type": "function", "function": {"name": "create_graph"}},
            temperature=0.2,
        )
        return Graph.model_validate(self._extract_fn_call(response))

    def _check_completeness(self, user_prompt: str, graph: Graph) -> str:
        node_list = "\n".join(
            f"  - id={n.id}, description={n.description!r}, "
            f"capability={n.capability}, depends_on={n.depends_on}"
            for n in graph.nodes
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a completeness auditor. Given a user job and a "
                    "proposed task graph, check whether every distinct entity "
                    "or subtask explicitly named in the job is COVERED by at "
                    "least one node in the graph.\n\n"
                    "COVERAGE RULES (apply these strictly):\n"
                    "- A vision node for an image = the image is covered.\n"
                    "- A web_search node mentioning an entity = that entity "
                    "is researched/analyzed.\n"
                    "- A summarization node for an entity = that entity is "
                    "summarized/analyzed.\n"
                    "- Any node whose description mentions an entity covers "
                    "that entity, regardless of capability type.\n"
                    "- A chain like vision -> web_search -> summarization for "
                    "an entity = fully covered.\n\n"
                    "Only respond INCOMPLETE if an entity or subtask named in "
                    "the job has ZERO nodes mentioning it. Do NOT require "
                    "extra 'analysis', 'deep dive', or 'dedicated analysis' "
                    "nodes beyond what already exists.\n\n"
                    "Respond with EXACTLY one of:\n"
                    "- COMPLETE\n"
                    "- INCOMPLETE: <short reason>\n\n"
                    "One line only."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"JOB: {user_prompt}\n\n"
                    f"GRAPH NODES:\n{node_list}"
                ),
            },
        ]
        response = self._client.chat.completions.create(
            model=_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=200,
        )
        text = (response.choices[0].message.content or "").strip()
        if text.upper().startswith("COMPLETE"):
            return "COMPLETE"
        return text

    def _call_llm_with_completeness_retry(
        self, user_prompt: str, inputs: Optional[dict[str, str]], reason: str
    ) -> Graph:
        messages = self._build_messages(user_prompt, inputs)
        messages.append(
            {
                "role": "user",
                "content": (
                    "Your previous graph was flagged as incomplete:\n"
                    f"{reason}\n\n"
                    "Please regenerate the graph ensuring every distinct entity "
                    "or subtask named in the job has its own dedicated node(s). "
                    "Do NOT combine multiple named entities into a single node. "
                    "Call create_graph with the corrected graph."
                ),
            }
        )
        response = self._client.chat.completions.create(
            model=_MODEL,
            messages=messages,
            tools=[_CREATE_GRAPH_TOOL],
            tool_choice={"type": "function", "function": {"name": "create_graph"}},
            temperature=0.2,
        )
        return Graph.model_validate(self._extract_fn_call(response))

    @staticmethod
    def _build_messages(
        user_prompt: str, inputs: Optional[dict[str, str]] = None
    ) -> list:
        messages: list = [
            {"role": "system", "content": _SYSTEM_PROMPT},
        ]

        text = user_prompt
        if inputs:
            from pathlib import Path as _P

            input_hints = []
            for input_type, input_path in sorted(inputs.items()):
                if not _P(input_path).exists():
                    raise FileNotFoundError(f"{input_type} file not found: {input_path}")
                label = input_type.capitalize()
                input_hints.append(f"[User has provided a {label} file at: {input_path}]")
            text = f"{user_prompt}\n\n" + "\n".join(input_hints)

        messages.append({"role": "user", "content": text})

        return messages

    @staticmethod
    def _extract_fn_call(response) -> dict:
        message = response.choices[0].message
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            return json.loads(tool_call.function.arguments)
        raise RuntimeError(
            "LLM did not return a create_graph function call. "
            f"Content: {message.content}"
        )

    @staticmethod
    def write_plan(graph: Graph) -> None:
        """Public entry point so main.py can re-render plan.md after DNA extraction.

        The plan is written once at decomposition time (no DNA yet) and again
        after the extractor runs, so the artifact on disk always reflects the
        graph as it will actually be scheduled.
        """
        ManagerAgent._write_plan_md(graph)

    @staticmethod
    def _write_plan_md(graph: Graph) -> None:
        waves = build_waves(graph)

        # --- node table ---
        lines = [
            "# Task Plan",
            f"**Job:** {graph.job}",
            "",
            "| Node ID | Description | Capability | DNA Flags | Demand | Cost Ceiling | Latency SLO | Depends On |",
            "|---------|-------------|------------|-----------|--------|--------------|-------------|------------|",
        ]
        for node in graph.nodes:
            deps = ", ".join(node.depends_on) if node.depends_on else "-"
            dna = node.dna
            if dna:
                flags = ", ".join(f"`{f}`" for f in dna.flags) if dna.flags else "-"
                demand = f"{dna.ordinals.demand():.2f}"
                ceiling = f"${dna.constraints.cost_ceiling_usd:.4f}"
                slo = f"{dna.constraints.latency_slo_ms}ms"
            else:
                flags, demand, ceiling, slo = "-", "-", "-", "-"
            lines.append(
                f"| {node.id} | {node.description} | {node.capability} "
                f"| {flags} | {demand} | {ceiling} | {slo} | {deps} |"
            )

        # --- per-node DNA detail, only once the extractor has run ---
        extracted = [n for n in graph.nodes if n.dna]
        if extracted:
            lines += ["", "## Capability DNA (per subtask)", ""]
            for node in extracted:
                dna = node.dna
                ords = ", ".join(
                    f"{k}={v}" for k, v in dna.ordinals.model_dump().items()
                )
                lines += [
                    f"- **`{node.id}`** — flags `{dna.flags}`",
                    f"  - ordinals: {ords}",
                    f"  - constraints: cost≤${dna.constraints.cost_ceiling_usd:.4f}, "
                    f"latency≤{dna.constraints.latency_slo_ms}ms, "
                    f"min_quality≥{dna.constraints.min_quality:.2f}, "
                    f"risk≤{dna.constraints.risk_tolerance}",
                    f"  - effective min_quality after ordinals: "
                    f"{dna.effective_min_quality():.2f}",
                    f"  - extracted by `{dna.extracted_by}` "
                    f"(confidence {dna.confidence:.2f})",
                ]

        # --- wave breakdown ---
        wave_parts = []
        for wave_idx, wave in enumerate(waves):
            ids = ", ".join(n.id for n in wave)
            wave_parts.append(f"Wave {wave_idx}: {ids}")
        lines += [
            "",
            "**Waves:** " + " | ".join(wave_parts),
            "",
        ]

        # --- mermaid diagram (deterministic, not LLM-generated) ---
        mermaid = build_mermaid(graph, waves)
        lines += [
            "```mermaid",
            mermaid,
            "```",
        ]

        path = Path(__file__).resolve().parent.parent / "outputs" / "plan.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _ensure_readme() -> None:
        path = Path(__file__).resolve().parent.parent / "README.md"
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        # Match the v0.0.2 heading too, so an upgraded repo replaces the stale
        # section instead of appending a second one below it.
        markers = [
            "## How AOS v0.0.3 plans and executes tasks",
            "## How AOS v0.0.2 plans and executes tasks",
        ]
        found = next((m for m in markers if m in existing), None)
        if found:
            start = existing.index(found)
            content = existing[:start].rstrip("\n") + "\n\n" + _README_SECTION
        elif existing:
            content = existing.rstrip("\n") + "\n\n" + _README_SECTION
        else:
            content = _README_SECTION
        path.write_text(content, encoding="utf-8")

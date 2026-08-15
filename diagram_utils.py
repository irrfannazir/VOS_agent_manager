from typing import List

from models import Graph, Node


def _sanitize(text: str, max_len: int = 40) -> str:
    """Strip/escape characters that break Mermaid syntax, truncate to max_len."""
    text = text.replace('"', "'").replace("[", "(").replace("]", ")")
    text = text.replace("|", "-").replace("{", "(").replace("}", ")")
    text = text.replace("\n", " ").strip()
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return text


_CAPABILITY_COLORS = {
    "web_search": "#4CAF50",
    "summarization": "#2196F3",
    "vision": "#FF9800",
    "synthesis": "#9C27B0",
}


def build_mermaid(graph: Graph, waves: List[List[Node]]) -> str:
    """Return a valid Mermaid flowchart string grouped by wave with classDef styling."""
    lines: list[str] = ["flowchart TD"]

    # --- subgraphs per wave ---
    for wave_idx, wave in enumerate(waves):
        lines.append(f"    subgraph Wave {wave_idx}")
        for node in wave:
            label = _sanitize(node.description)
            lines.append(f'        {node.id}["{label} ({node.capability})"]')
        lines.append("    end")

    # --- dependency edges ---
    id_set = {n.id for n in graph.nodes}
    for node in graph.nodes:
        for dep in node.depends_on:
            if dep in id_set:
                lines.append(f"    {dep} --> {node.id}")

    # --- classDef + class assignments ---
    lines.append("")
    for cap, color in _CAPABILITY_COLORS.items():
        lines.append(f"    classDef {cap}Style fill:{color},color:#fff")
    for node in graph.nodes:
        style = _CAPABILITY_COLORS.get(node.capability, "")
        if style:
            lines.append(f"    class {node.id} {node.capability}Style")

    return "\n".join(lines)

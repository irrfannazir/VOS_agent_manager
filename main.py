import re
import sys
from pathlib import Path

# The root of this repo IS the package, so it must be importable by plain module
# name before the aos_v0 shim can alias anything.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bootstrap import install_aos_v0_shim

install_aos_v0_shim()

from agents.graph_executor import GraphExecutor
from agents.integrator_agent import IntegratorAgent
from agents.manager_agent import ManagerAgent
from constraint_policy import ConstraintPolicy
from dna_extractor import DNAExtractor
from failure_manager import FailureManager
from resource_registration import build_default_registry

DEFAULT_BUDGET_USD = 0.50


def run(
    user_prompt: str,
    image_path: str | None = None,
    budget_usd: float = DEFAULT_BUDGET_USD,
) -> str:
    registry = build_default_registry()

    # M2 -- decomposition only. The planner no longer reasons about resources,
    # and the kernel appends the terminal synthesis node itself.
    manager = ManagerAgent()
    graph = manager.create_plan(user_prompt, image_path)

    # M4 -- Capability DNA extraction: flags + ordinals only.
    graph = DNAExtractor().extract_graph(graph)

    # Constraints are kernel-derived, never model-invented. Budget is divided
    # across the plan, so a wide graph automatically routes cheaper.
    ConstraintPolicy(registry, job_budget_usd=budget_usd).apply(graph)

    # Admission control seed (DOC1 M2): reject before spending anything if the
    # registry simply cannot provide a capability the plan requires.
    _check_satisfiable(graph, registry)

    # Re-render the plan artifact now that every node carries its full DNA.
    manager.write_plan(graph)

    # M5 routing + M8 recovery happen per node, inside the executor.
    failure_manager = FailureManager(registry)
    graph = GraphExecutor(registry, failure_manager).run(graph, image_path)

    final_output = IntegratorAgent().integrate(graph)

    _print_routing_summary(graph)
    print("\n" + failure_manager.report())

    print("\n=== FINAL OUTPUT ===")
    print(final_output)
    return final_output


def _check_satisfiable(graph, registry) -> None:
    """Fail fast, naming the missing capability, before any node executes."""
    problems = []
    for node in graph.nodes:
        if not node.dna:
            continue
        missing = registry.unsatisfiable_flags(node.dna)
        if missing:
            problems.append(f"node '{node.id}' requires {missing}")
    if problems:
        raise RuntimeError(
            "[admission-control] plan rejected — no registered resource provides: "
            + "; ".join(problems)
        )
    print("[admission-control] all DNA flags satisfiable by registered resources")


def _print_routing_summary(graph) -> None:
    print("\n=== ROUTING SUMMARY ===")
    dna_routed = 0
    for node in graph.nodes:
        flags = ", ".join(node.dna.flags) if node.dna and node.dna.flags else "-"
        if node.routing_mode == "dna":
            dna_routed += 1
        print(
            f"  {node.id:<8} {node.routing_mode or '-':<6} {node.status:<9} -> "
            f"{node.bound_resource or '-':<22} flags=[{flags}]"
        )
    print(f"  {dna_routed}/{len(graph.nodes)} nodes bound by Capability DNA")


def _extract_image(text: str) -> tuple[str, str | None]:
    """Extract --image <path> from text, returning cleaned text and path."""
    match = re.search(r"--image\s+(\S+)", text)
    if match:
        image_path = match.group(1)
        cleaned = text[: match.start()] + text[match.end() :]
        return cleaned.strip(), image_path
    return text, None


def _pop_flag(args: list[str], flag: str) -> str | None:
    """Remove `flag <value>` from args, returning the value."""
    if flag not in args:
        return None
    idx = args.index(flag)
    if idx + 1 >= len(args):
        print(f"Error: {flag} requires a value")
        sys.exit(1)
    value = args[idx + 1]
    del args[idx : idx + 2]
    return value


if __name__ == "__main__":
    args = sys.argv[1:]

    image = _pop_flag(args, "--image")
    budget_arg = _pop_flag(args, "--budget")
    budget = float(budget_arg) if budget_arg else DEFAULT_BUDGET_USD

    prompt = " ".join(args) if args else input("Enter your prompt: ")

    prompt, image_from_text = _extract_image(prompt)
    if image_from_text:
        image = image_from_text

    if not prompt:
        print("Error: no prompt provided")
        sys.exit(1)

    run(prompt, image, budget)

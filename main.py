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
from resource_registration import build_default_registry, build_hf_enabled_registry

DEFAULT_BUDGET_USD = 0.50

# File extensions recognised for auto-detection.
_AUDIO_EXTS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".wma", ".aac"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"}
_TEXT_EXTS = {".txt", ".md", ".csv", ".json", ".pdf"}


def run(
    user_prompt: str,
    inputs: dict[str, str] | None = None,
    budget_usd: float = DEFAULT_BUDGET_USD,
) -> str:
    registry = build_hf_enabled_registry()

    # M2 -- decomposition only. The planner no longer reasons about resources,
    # and the kernel appends the terminal synthesis node itself.
    manager = ManagerAgent()
    graph = manager.create_plan(user_prompt, inputs)

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
    graph = GraphExecutor(registry, failure_manager).run(graph, inputs)

    final_output = IntegratorAgent().integrate(graph)

    _print_routing_summary(graph)
    print("\n" + failure_manager.report())

    print("\n=== FINAL OUTPUT ===")
    try:
        print(final_output)
    except UnicodeEncodeError:
        print(final_output.encode("ascii", "replace").decode("ascii"))
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
    relaxed_routed = 0
    for node in graph.nodes:
        flags = ", ".join(node.dna.flags) if node.dna and node.dna.flags else "-"
        mode = node.routing_mode or "-"
        if mode == "dna":
            dna_routed += 1
        elif mode == "relaxed":
            relaxed_routed += 1
        status_tag = node.status.upper() if node.status in ("done", "degraded", "failed") else node.status
        print(
            f"  {node.id:<8} {mode:<8} {status_tag:<9} -> "
            f"{node.bound_resource or '-':<22} flags=[{flags}]"
        )
    total = len(graph.nodes)
    print(
        f"  {dna_routed}/{total} exact DNA, "
        f"{relaxed_routed}/{total} relaxed (degraded), "
        f"{total - dna_routed - relaxed_routed}/{total} other"
    )


def _detect_input_type(path: str) -> str:
    """Auto-detect input type from file extension."""
    ext = Path(path).suffix.lower()
    if ext in _AUDIO_EXTS:
        return "audio"
    if ext in _IMAGE_EXTS:
        return "image"
    if ext in _TEXT_EXTS:
        return "text"
    return "text"


def _scan_inputs_folder() -> dict[str, str]:
    """Scan the inputs/ folder for files with known extensions.

    Returns {type: path} for each found file. If multiple files of the same
    type exist, only the first one is used (subsequent calls would need a list
    API, which is out of scope for now).
    """
    inputs_dir = Path(__file__).resolve().parent / "inputs"
    if not inputs_dir.is_dir():
        return {}

    found: dict[str, str] = {}
    for f in sorted(inputs_dir.iterdir()):
        if f.is_file() and not f.name.startswith("."):
            input_type = _detect_input_type(str(f))
            if input_type not in found:
                found[input_type] = str(f)
    return found


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


def _collect_inputs(args: list[str]) -> dict[str, str]:
    """Collect typed inputs from CLI flags and the inputs/ folder.

    Priority: explicit CLI flags (--input, --audio, --image) override the
    inputs/ folder scan. Returns {type: path}.
    """
    inputs: dict[str, str] = {}

    # 1. Folder scan (lowest priority).
    inputs.update(_scan_inputs_folder())

    # 2. --input <path> (auto-detect type from extension, repeatable).
    while "--input" in args:
        idx = args.index("--input")
        if idx + 1 >= len(args):
            print("Error: --input requires a file path")
            sys.exit(1)
        path = args[idx + 1]
        del args[idx : idx + 2]
        if not Path(path).exists():
            print(f"Error: input file not found: {path}")
            sys.exit(1)
        input_type = _detect_input_type(path)
        inputs[input_type] = path

    # 3. --audio <path> (shorthand for --input with audio type).
    while "--audio" in args:
        idx = args.index("--audio")
        if idx + 1 >= len(args):
            print("Error: --audio requires a file path")
            sys.exit(1)
        path = args[idx + 1]
        del args[idx : idx + 2]
        if not Path(path).exists():
            print(f"Error: audio file not found: {path}")
            sys.exit(1)
        inputs["audio"] = path

    # 4. --image <path> (legacy flag, still supported).
    while "--image" in args:
        idx = args.index("--image")
        if idx + 1 >= len(args):
            print("Error: --image requires a file path")
            sys.exit(1)
        path = args[idx + 1]
        del args[idx : idx + 2]
        if not Path(path).exists():
            print(f"Error: image file not found: {path}")
            sys.exit(1)
        inputs["image"] = path

    return inputs


if __name__ == "__main__":
    args = sys.argv[1:]

    budget_arg = _pop_flag(args, "--budget")
    budget = float(budget_arg) if budget_arg else DEFAULT_BUDGET_USD

    # Collect all typed inputs (--input, --audio, --image, inputs/ folder).
    inputs = _collect_inputs(args)

    prompt = " ".join(args) if args else input("Enter your prompt: ")

    # Also accept --image embedded in the prompt text (backward compat).
    prompt, image_from_text = _extract_image(prompt)
    if image_from_text:
        inputs["image"] = image_from_text

    if not prompt:
        print("Error: no prompt provided")
        sys.exit(1)

    run(prompt, inputs if inputs else None, budget)

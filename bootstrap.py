"""Import shim for the legacy `aos_v0.*` package paths.

Modules under agents/ and capabilities/ still import `from aos_v0.X import ...`
from the v0.0.2 package layout, but this repo has no `aos_v0` package -- the
root IS the package. Rather than rewrite every import, alias the already
imported root modules into an `aos_v0` namespace.

The aliasing matters more than it looks: mapping `aos_v0.models` to the *same
module object* as root `models` means there is exactly one CapabilityDNA class.
Giving the shim a real __path__ instead would let Python import a second copy
from disk, and every isinstance/Pydantic check across the boundary would fail.
"""

import importlib
import sys
import types

# Order matters only in that each of these must import cleanly on its own.
_ROOT_MODULES = [
    "config",
    "models",
    "graph_utils",
    "diagram_utils",
    "capability_registry",
    "constraint_policy",
    "failure_manager",
    "dna_extractor",
]

_AGENT_MODULES = [
    "sub_agent",
    "graph_executor",
    "manager_agent",
    "integrator_agent",
]


def install_aos_v0_shim() -> None:
    """Idempotently alias root modules under the `aos_v0` namespace."""
    if "aos_v0" in sys.modules:
        return

    pkg = types.ModuleType("aos_v0")
    # Empty __path__: this is a namespace alias, never a real import root.
    pkg.__path__ = []
    sys.modules["aos_v0"] = pkg

    for name in _ROOT_MODULES:
        module = importlib.import_module(name)
        sys.modules[f"aos_v0.{name}"] = module
        setattr(pkg, name, module)

    # agents/* import aos_v0.models etc. at module scope, so the root aliases
    # above must already be in place before this import runs.
    agents_pkg = importlib.import_module("agents")
    sys.modules["aos_v0.agents"] = agents_pkg
    setattr(pkg, "agents", agents_pkg)

    for name in _AGENT_MODULES:
        module = importlib.import_module(f"agents.{name}")
        sys.modules[f"aos_v0.agents.{name}"] = module
        setattr(agents_pkg, name, module)

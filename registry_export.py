"""Export the resource registry to a machine-readable JSON file.

Produces ``outputs/registry_data.json`` containing every registered resource
for the three pools the system can construct:

  - ``hf``        the HF Inference Providers catalog (the expanded model set)
  - ``default``   the frozen DOC2 default pool
  - ``mixed``     default pool + HF catalog (the auto-router view)

Entries are ``CapabilityManifest.model_dump()`` so the file uses the same
schema as the in-memory registry. Deterministic: keys sorted, stable ordering.
Run with ``python registry_export.py``.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bootstrap import install_aos_v0_shim

install_aos_v0_shim()

from capability_registry import CapabilityRegistry
from resource_registration import (
    build_default_registry,
    build_hf_enabled_registry,
)
import providers.hf as hf_mod

OUT = Path(__file__).resolve().parent / "outputs" / "registry_data.json"


def _pool(name: str, registry: CapabilityRegistry) -> dict:
    manifests = [m.model_dump() for m in registry.manifests()]
    return {
        "name": name,
        "resource_count": len(manifests),
        "distinct_capability_flags": sorted(
            registry.provided_flags()
        ),
        "resources": manifests,
    }


def main() -> int:
    hf_only = CapabilityRegistry()
    hf_mod.register_hf_resources(hf_only)

    data = {
        "schema": "aosv0.0.3/resource-registry",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pools": {
            "hf": _pool("hf-inference-providers-catalog", hf_only),
            "default": _pool("default-doc2-pool", build_default_registry()),
            "mixed": _pool("hf-enabled-registry", build_hf_enabled_registry()),
        },
    }

    payload = json.dumps(data, indent=2, sort_keys=True)
    OUT.write_text(payload, encoding="utf-8")
    print(f"wrote {OUT}")
    for name, pool in data["pools"].items():
        print(f"  {name:8s} {pool['resource_count']:2d} resources  "
              f"{len(pool['distinct_capability_flags']):2d} distinct flags")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Provider adapters.

Each provider is a self-contained module that plugs into the Capability
Registry as ordinary (CapabilityManifest, run_fn) pairs -- the core never
imports a provider directly. `providers.hf` is the Hugging Face Inference
Providers adapter; future providers follow the same shape.
"""

from providers.errors import (
    ProviderAuthenticationError,
    ProviderBadRequestError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from providers.hf import (
    DEFAULT_HF_MODEL,
    HF_MODEL_CATALOG,
    HFProvider,
    HFResult,
    build_hf_manifests,
    register_hf_resources,
    run,
    run_result,
)

__all__ = [
    "DEFAULT_HF_MODEL",
    "HF_MODEL_CATALOG",
    "HFProvider",
    "HFResult",
    "ProviderAuthenticationError",
    "ProviderBadRequestError",
    "ProviderError",
    "ProviderRateLimitError",
    "ProviderTimeoutError",
    "ProviderUnavailableError",
    "build_hf_manifests",
    "register_hf_resources",
    "run",
    "run_result",
]

"""Typed errors raised by provider adapters.

These are provider-domain errors, not core-model errors: the capability system
never sees them at decision time (the Failure Manager catches whatever a run_fn
raises and classifies it). Their job is to make a provider failure meaningful at
the boundary -- authentication vs. rate limit vs. model-down vs. timeout -- so
callers can react differently to each.

Contract: ProviderError messages MUST NOT contain credentials. Adapters redact
before raising.
"""


class ProviderError(RuntimeError):
    """Base class for all provider errors. Messages never carry secrets."""


class ProviderAuthenticationError(ProviderError):
    """Missing or invalid credentials (e.g. no HF_TOKEN, or a rejected token)."""


class ProviderRateLimitError(ProviderError):
    """The provider throttled the request (HTTP 429)."""


class ProviderTimeoutError(ProviderError):
    """The provider did not answer within the timeout window."""


class ProviderBadRequestError(ProviderError):
    """The request itself is invalid -- unsupported model/task, HTTP 4xx."""


class ProviderUnavailableError(ProviderError):
    """The model or provider is unavailable (HTTP 5xx, inference server down,
    or a network/connection failure)."""

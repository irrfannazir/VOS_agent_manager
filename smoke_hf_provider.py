"""Manual Hugging Face Inference Providers smoke test.

OPTIONAL / MANUAL -- this is NOT part of the normal test suite. It makes ONE
real (paid) inference call, so it must be run explicitly and deliberately:

    python smoke_hf_provider.py

It is intentionally not named test_*.py, has no auto-discovery hooks, and does
nothing when imported -- it only runs under `__main__`.

Setup (local):

    1. Get an HF token with "Make calls to Inference Providers" permission:
       https://huggingface.co/settings/tokens
    2. Copy .env.example to .env (or export the variables in your shell):
         HF_TOKEN=your_token_here
         HF_PROVIDER=auto                # optional: auto, together, nebius, ...
         HF_MODEL=                       # optional: default model id
    3. Install dependencies:  pip install -r requirements.txt
    4. Run:  python smoke_hf_provider.py

Cost: a single, minimal deterministic request (temperature=0, max_tokens=16),
so the inference cost is negligible. The token is read from the environment and
is NEVER printed or logged.

Exit codes: 0 = success, 1 = inference failed, 2 = HF_TOKEN missing.
"""

import os
import sys

from dotenv import load_dotenv


def _fail(message: str, code: int) -> int:
    print(f"[smoke] ERROR: {message}", file=sys.stderr)
    return code


def main() -> int:
    # 1) Read HF_TOKEN from the environment (a local .env is fine too).
    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    if not token:
        return _fail(
            "HF_TOKEN is not set. Create a token at "
            "https://huggingface.co/settings/tokens (enable 'Make calls to "
            "Inference Providers'), then run with it exported or in .env "
            "(copy .env.example to .env). See smoke_hf_provider.py header.",
            2,
        )

    # Never print the token: this helper is the only thing that writes to
    # stdout/stderr for user-facing output, and it refuses to emit it.
    def _safe_print(line: str) -> None:
        if token and token in line:
            raise RuntimeError("refusing to print a string containing the HF token")
        print(line)

    # 2/3) Configured HF model (HF_MODEL env) with a built-in fallback.
    from providers.hf import DEFAULT_HF_MODEL, HFProvider

    model = os.environ.get("HF_MODEL") or DEFAULT_HF_MODEL
    provider = os.environ.get("HF_PROVIDER") or "auto"

    _safe_print(f"[smoke] model:    {model}")
    _safe_print(f"[smoke] provider: {provider}")

    # One minimal deterministic request.
    try:
        result = HFProvider(token=token, model=model, provider=provider).execute(
            messages=[{"role": "user", "content": "Reply with exactly: PONG"}],
            temperature=0.0,
            max_tokens=16,
        )
    except Exception as exc:  # noqa: BLE001 -- exit cleanly with a readable error
        # The adapter already redacts the token from ProviderError messages.
        return _fail(f"inference failed: {exc}", 1)

    # 4) The response.
    _safe_print(f"[smoke] response: {result.text}")

    # 5) Non-secret metadata, when available.
    _safe_print(f"[smoke] model:    {result.model or 'n/a'}")
    _safe_print(f"[smoke] provider: {result.provider or 'n/a'}")
    if result.usage:
        usage = result.usage
        _safe_print(
            f"[smoke] usage:    "
            f"prompt={usage.get('prompt_tokens', 'n/a')} "
            f"completion={usage.get('completion_tokens', 'n/a')} "
            f"total={usage.get('total_tokens', 'n/a')}"
        )
    else:
        _safe_print("[smoke] usage:    n/a")
    if result.raw is not None:
        completion_id = getattr(result.raw, "id", None)
        fingerprint = getattr(result.raw, "system_fingerprint", None)
        if completion_id:
            _safe_print(f"[smoke] request_id: {completion_id}")
        if fingerprint:
            _safe_print(f"[smoke] fingerprint: {fingerprint}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

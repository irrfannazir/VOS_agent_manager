import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")

# Hugging Face Inference Providers token. Prefer HF_TOKEN; fall back to the
# legacy `HUG` key already present in this repo's .env so nothing breaks.
HF_TOKEN: str = os.environ.get("HF_TOKEN") or os.environ.get("HUG", "")

# Inference Providers routing: "auto" lets Hugging Face pick a provider, or
# name one explicitly (e.g. "together", "nebius", "groq", ...).
HF_PROVIDER: str = os.environ.get("HF_PROVIDER", "auto")

# Default model id for HF-backed resources. Overridable per call / per resource.
HF_MODEL: str = os.environ.get("HF_MODEL", "")

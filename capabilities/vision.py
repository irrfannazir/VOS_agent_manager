import base64
from typing import Optional

from openai import OpenAI

_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

_MODEL = "medgemma1.5:latest"

_DEFAULT_PROMPT = "Describe this image in detail."


def run(image_path: str, instruction: Optional[str] = None) -> str:
    prompt = instruction if instruction else _DEFAULT_PROMPT

    from pathlib import Path

    if not Path(image_path).is_file():
        raise FileNotFoundError(
            f"Vision capability requires a valid image file, "
            f"got: '{image_path}'\n"
            f"Run with: python main.py \"prompt\" --image path/to/image.jpg"
        )

    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    import mimetypes

    mime, _ = mimetypes.guess_type(image_path)
    data_url = f"data:{mime or 'image/png'};base64,{image_data}"

    response = _client.chat.completions.create(
        model=_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ],
            }
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""

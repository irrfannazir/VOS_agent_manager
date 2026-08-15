from typing import Optional

from groq import Groq

from aos_v0.config import GROQ_API_KEY

_client = Groq(api_key=GROQ_API_KEY)

_MODEL = "llama-3.3-70b-versatile"

_DEFAULT_SYSTEM = (
    "You are a factual summarizer. Given a body of text, extract and "
    "preserve ALL specific facts, numbers, dates, statistics, and names. "
    "Produce a detailed summary that keeps every concrete data point. "
    "Never use vague language like 'multiple', 'not specified', or 'various' "
    "when exact numbers are available in the source text.\n\n"
    "PRESERVE ENUMERATION: if the source contains a list of items (news "
    "stories, match results, entities), keep every item as its own entry. "
    "Never collapse a list of N items into a sentence describing the list. "
    "If the source says there were five news stories, your output has five "
    "numbered entries.\n\n"
    "If the source text contains no usable information, say exactly "
    "'NO_DATA: <what was missing>' and nothing else, so the kernel can detect "
    "the gap instead of inheriting an empty summary.\n\n"
    "Return only the summary text with no preamble."
)


def run(text: str, instruction: Optional[str] = None) -> str:
    """`instruction` is the node description. It AUGMENTS the system prompt.

    It used to replace it outright, which silently discarded every
    fact-preservation rule above the moment a node had a description -- i.e.
    always. That is the single biggest reason early runs returned vague,
    compressed output.
    """
    system = _DEFAULT_SYSTEM
    if instruction:
        system = f"{_DEFAULT_SYSTEM}\n\nTHIS SUBTASK: {instruction}"

    response = _client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
        temperature=0.3,
        max_tokens=2048,
    )
    return response.choices[0].message.content or ""

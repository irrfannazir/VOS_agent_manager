from typing import Optional

from ddgs import DDGS
from groq import Groq

from aos_v0.config import GROQ_API_KEY

_client = Groq(api_key=GROQ_API_KEY)
_MODEL = "llama-3.3-70b-versatile"

# Sentinel the Failure Manager's content detector looks for. Emitting a token
# instead of prose means an empty result is a *classifiable* failure rather
# than a plausible-looking answer that quietly poisons everything downstream.
NO_DATA_PREFIX = "NO_DATA:"

_SYSTEM = (
    "You are a research assistant. Given web search results, extract ALL "
    "specific facts, numbers, dates, statistics, and data points. Be extremely "
    "precise — include exact numbers (e.g. '5 World Cup titles', '1966', "
    "'1-0').\n\n"
    "PRESERVE ENUMERATION: if the query asks for a number of items (5 news "
    "stories) or for 'all' of something (every match result), return them as a "
    "numbered list with one entry per item, each with its own date, names and "
    "figures. Never collapse a list into a summary sentence about the list.\n\n"
    f"If the search results genuinely do not answer the query, reply with "
    f"'{NO_DATA_PREFIX} <what was missing>' and nothing else. Do not guess, do "
    "not substitute related information, and do not pad. A clean miss is "
    "recoverable; a fabricated answer is not.\n\n"
    "Return only the factual answer with no preamble."
)


def run(query: str, instruction: Optional[str] = None) -> str:
    """`instruction` is the node description; `query` is the upstream input.

    The description is the better search query -- it names the entity the
    planner isolated -- but the raw input still carries context the extraction
    pass needs, so both are passed on rather than one replacing the other.
    """
    search_query = instruction if instruction else query

    results_text = _search_web(search_query)
    if not results_text:
        return f"{NO_DATA_PREFIX} web search returned no results for {search_query!r}"

    response = _client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": (
                    f"QUERY: {search_query}\n\n"
                    f"WEB SEARCH RESULTS:\n{results_text}"
                ),
            },
        ],
        temperature=0.3,
        max_tokens=2048,
    )
    return response.choices[0].message.content or f"{NO_DATA_PREFIX} empty model reply"


def _search_web(query: str, max_results: int = 8) -> str:
    """Raised from 5 to 8 results: queries asking for 'all matches' or '5 news
    items' were starving on 5 hits, which read downstream as a data gap."""
    try:
        results = DDGS().text(query, max_results=max_results)
        if not results:
            return ""
        parts = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            body = r.get("body", "")
            parts.append(f"[{i}] {title}\n{body}")
        return "\n\n".join(parts)
    except Exception as exc:
        # Surfaced as text rather than raised: the Failure Manager classifies
        # this as a resource fault and decides whether to retry or substitute.
        return f"[Search error: {exc}]"

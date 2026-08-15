"""Final-answer synthesis.

Distinct from summarization on purpose. The summarizer's job is to compress;
this one's job is to ANSWER. The first live run showed why the distinction
matters: a job asking for "5 news items" and "all Argentina match results" was
routed through summarizers at every stage, so the user got a compressed digest
of a digest instead of the enumerated detail they asked for.

So this capability is instructed to expand rather than compress: it re-reads
the user's original job, answers every part of it separately, and preserves
every enumerated item and concrete data point the upstream nodes gathered.
"""

from typing import Optional

from groq import Groq

from aos_v0.config import GROQ_API_KEY

_client = Groq(api_key=GROQ_API_KEY)

_MODEL = "llama-3.3-70b-versatile"

_SYSTEM = """\
You write the FINAL ANSWER for a user's request, using research gathered by
other agents.

Your job is to ANSWER COMPLETELY, not to summarize. Follow these rules:

1. ADDRESS EVERY PART. The user's request may contain several distinct asks.
   Answer each one under its own clear heading, in the order the user asked.
   Never merge two separate asks into one paragraph.

2. RESPECT REQUESTED QUANTITIES. If the user asked for 5 items, give exactly 5
   numbered items. If they asked for "all" of something, list every one you
   have. Never reduce a requested list to a sentence about the list.

3. KEEP ALL SPECIFICS. Preserve every name, number, date, score, statistic and
   proper noun from the research. Do not round, generalise or drop detail. If
   the research says "defeated Argentina 1-0 on July 19, 2026", say exactly
   that.

4. BE HONEST ABOUT GAPS. If the research does not cover one of the user's
   asks, say so plainly under that ask's heading -- one line, stating what is
   missing and why (e.g. "the search returned no match-by-match results").
   Never invent facts to fill a gap, and never let one missing ask shorten
   your answer to the others.

5. FLAG UNVERIFIED CLAIMS. If the research contains something that looks like
   a future or unverifiable event, report it and mark it as unverified rather
   than dropping it.

Write in clear Markdown with headings and lists. Length should match the
detail available -- do not pad, but never compress to be brief."""


def run(text: str, instruction: Optional[str] = None) -> str:
    """`text` is the concatenated upstream research; `instruction` carries the
    node description, which by convention embeds the user's original job."""
    user_content = text
    if instruction:
        user_content = (
            f"USER'S REQUEST (answer every part of this):\n{instruction}\n\n"
            f"---\n\nRESEARCH GATHERED BY OTHER AGENTS:\n{text}"
        )

    response = _client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
        # Generous ceiling: this node is the one place where truncating the
        # output directly costs the user the detail they asked for.
        max_tokens=4096,
    )
    return response.choices[0].message.content or ""

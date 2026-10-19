"""Goal: generic, data-driven hygiene on raw curriculum skills (no LLM, no domain lists):
encoding repair, whitespace, Title Case (source acronyms kept), case-insensitive dedup.

Example:
  ["  supervised learning ", "Supervised Learning", "differentâ¦", "AI Sub-Fields"]
  => ["Supervised Learning", "Different…", "AI Sub-Fields"]
"""

import re
import unicodedata

try:
    from ftfy import fix_text
except ImportError:
    def fix_text(s: str) -> str:
        return unicodedata.normalize("NFKC", s)

_SMALL = {"a", "an", "and", "as", "at", "but", "by", "for", "if", "in", "of",
          "on", "or", "the", "to", "via", "vs", "with", "nor", "per"}


def clean_skills(skills: list[str]) -> list[str]:
    def _case(name: str) -> str:
        words = []
        for i, tok in enumerate(name.split()):

            if any(c.isalpha() for c in tok) and tok == tok.upper() and len(tok) <= 5:
                words.append(tok)
            elif i and tok.lower() in _SMALL:
                words.append(tok.lower())
            else:
                words.append("-".join(p[:1].upper() + p[1:].lower()
                                      for p in tok.split("-")))
        return " ".join(words)

    seen, out = set(), []
    for raw in skills:
        if not raw:
            continue
        name = fix_text(raw)
        name = re.sub(r"\s+", " ", name).strip(" \t\"'·–—-")
        if not name:
            continue
        name = _case(name)
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(name)
    return out


if __name__ == "__main__":
    raw = ["  supervised learning ", "Supervised Learning", "K-Nearest Neighbor",
           "Data Exploration and Analysis (EDA)", "differentâ¦",
           "acid-base titration"]
    print(clean_skills(raw))

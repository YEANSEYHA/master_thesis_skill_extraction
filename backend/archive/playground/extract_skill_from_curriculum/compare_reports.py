"""Compare every report_*.md against the reference model, per course.

Each report embeds its raw JSON, so this reads the reports themselves — no API
calls. Skill names are matched semantically-ish (normalised token sets, aliases,
containment), because "K-Means" and "K-Means Clustering" are the same skill.

    python compare_reports.py            # reference = gpt55
    python compare_reports.py glm        # reference = report_glm.md
"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
STOP = {"of", "the", "and", "a", "an", "for", "to", "in", "with", "its", "from"}

ALIAS = {
    "cnn": "convolutional neural network",
    "pca": "principal component analysis",
    "svm": "support vector machine",
    "mlp": "multi layer perceptron",
    "ai": "artificial intelligence",
    "nlp": "natural language processing",
    "pmi": "pointwise mutual information",
    "knn": "k nearest neighbor",
    "tf idf": "tfidf",
}


def toks(s: str) -> set:
    s = re.sub(r"\(.*?\)", " ", s.lower())
    s = ALIAS.get(re.sub(r"[^a-z0-9]+", " ", s).strip(), s)
    return {w for w in re.findall(r"[a-z0-9]+", s) if w not in STOP}


def same(a: str, b: str) -> bool:
    ta, tb = toks(a), toks(b)
    if not ta or not tb:
        return False
    if ta == tb:
        return True
    inter = ta & tb
    if inter == ta or inter == tb:          # one is a qualified form of the other
        return True
    return len(inter) / len(ta | tb) >= 0.7


def prf(pred: list, gold: list):
    used, tp = set(), 0
    for p in pred:
        for i, g in enumerate(gold):
            if i not in used and same(p, g):
                used.add(i)
                tp += 1
                break
    P = tp / len(pred) if pred else 0.0
    R = tp / len(gold) if gold else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return P, R, F


def load(path: Path) -> dict:
    """report_*.md → {course_file: record} from the embedded raw JSON block."""
    m = re.search(r"## Raw JSON\s*```json\s*(.*?)```", path.read_text(), re.S)
    if not m:
        return {}
    return {r["file"]: r for r in json.loads(m.group(1))}


def names(rec) -> list:
    return [s["name"] for s in (rec or {}).get("skills", [])]


def main(ref_label="gpt55") -> None:
    reports = {p.stem.replace("report_", ""): load(p)
               for p in sorted(HERE.glob("report_*.md"))}
    if ref_label not in reports:
        raise SystemExit(f"no report_{ref_label}.md (have: {', '.join(reports)})")

    ref = reports[ref_label]
    others = [k for k in reports if k != ref_label]
    courses = sorted(ref)

    print(f"reference = {ref_label}   ({len(courses)} courses)\n")
    print(f"{'model':<16}{'courses':>8}{'skills':>8}{'P':>7}{'R':>7}{'F1':>7}"
          f"{'CLO Δ':>8}{'lesson Δ':>10}")
    print("-" * 71)

    rows = []
    for m in others:
        rep = reports[m]
        Ps, Rs, Fs, nsk, ok, dclo, dles = [], [], [], 0, 0, 0, 0
        for c in courses:
            rec = rep.get(c)
            if not rec or rec.get("error"):
                continue
            ok += 1
            nsk += len(names(rec))
            P, R, F = prf(names(rec), names(ref[c]))
            Ps.append(P); Rs.append(R); Fs.append(F)
            dclo += abs(len(rec.get("clos", [])) - len(ref[c].get("clos", [])))
            dles += abs(len(rec.get("lessons", [])) - len(ref[c].get("lessons", [])))
        n = len(Fs) or 1
        rows.append((sum(Fs) / n, m))
        print(f"{m:<16}{f'{ok}/{len(courses)}':>8}{nsk:>8}"
              f"{sum(Ps)/n:>7.2f}{sum(Rs)/n:>7.2f}{sum(Fs)/n:>7.2f}"
              f"{dclo:>8}{dles:>10}")

    print("\nranking by macro F1 vs reference:")
    for f, m in sorted(rows, reverse=True):
        print(f"  {f:.3f}  {m}")

    # where the biggest model disagrees most, and how
    print("\nper-course F1")
    hdr = f"{'course':<34}" + "".join(f"{m[:11]:>12}" for m in others)
    print(hdr); print("-" * len(hdr))
    for c in courses:
        cells = []
        for m in others:
            rec = reports[m].get(c)
            if not rec or rec.get("error"):
                cells.append(f"{'ERR':>12}"); continue
            cells.append(f"{prf(names(rec), names(ref[c]))[2]:>12.2f}")
        title = (ref[c].get("course") or {}).get("title") or c
        print(f"{title[:33]:<34}" + "".join(cells))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "gpt55")

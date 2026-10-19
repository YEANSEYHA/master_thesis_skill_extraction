"""Dynamic syllabus-to-job-field filtering for fair curriculum gap analysis."""

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_POSTINGS = HERE / "it_postings.jsonl"
TITLE_CACHE = HERE / ".job_title_embeddings_minilm.npz"
OUTPUT_DIR = HERE / "output"

PROFILE_MODEL = "qwen3:14b"
VERIFY_MODEL = "qwen3:4b"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


_PROFILE_SCHEMA = {
    "type": "object",
    "properties": {
        "course_title": {"type": "string"},
        "field_name": {"type": "string"},
        "field_description": {"type": "string"},
        "is_foundational": {"type": "boolean"},
        "target_roles": {
            "type": "array",
            "items": {"type": "string"},
        },
        "reason": {"type": "string"},
    },
    "required": [
        "course_title",
        "field_name",
        "field_description",
        "is_foundational",
        "target_roles",
        "reason",
    ],
}

_VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "is_related": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["index", "is_related", "reason"],
            },
        },
    },
    "required": ["decisions"],
}


def _read_pdf(pdf_path: str | Path) -> str:
    from pypdf import PdfReader

    path = Path(pdf_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Course document not found: {path}")
    if path.suffix.casefold() != ".pdf":
        raise ValueError(f"Expected a PDF course document: {path}")

    text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    if not text.strip():
        raise ValueError(f"No readable text found in: {path}")
    return text


def _course_profile(pdf_path: str | Path, document_text: str) -> dict:
    """Infer a field and role family from the document without a static taxonomy."""
    import ollama

    prompt_text = document_text[:50_000]
    response = ollama.chat(
        model=PROFILE_MODEL,
        think=False,
        options={"temperature": 0, "num_ctx": 16_384},
        format=_PROFILE_SCHEMA,
        messages=[
            {
                "role": "system",
                "content": (
                    "Read the university course document and identify the ONE "
                    "narrowest direct labour-market field supported by most of its "
                    "learning outcomes and lessons. Judge the actual description, "
                    "outcomes, and lessons; do not trust the filename or title when "
                    "the content disagrees. Return a precise field description and "
                    "6-10 real, commonly posted job-board titles for which this field "
                    "is the PRIMARY function and a defining competency. Never invent "
                    "a title by attaching 'specialist', 'analyst', or 'developer' to "
                    "a lesson topic. Do not include broad or adjacent jobs merely "
                    "because they use one transferable topic from the course. Titles "
                    "must be occupations, not skills, tools, degrees, industries, or "
                    "generic titles. Exclude adjacent legal, sales, academic, and "
                    "management-only roles. If the document is theoretical or "
                    "foundation knowledge without a direct occupation, set "
                    "is_foundational=true and target_roles=[]."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Uploaded filename: {Path(pdf_path).name}\n\n"
                    f"COURSE DOCUMENT:\n{prompt_text}"
                ),
            },
        ],
    )
    profile = json.loads(response.message.content)
    profile["target_roles"] = list(
        dict.fromkeys(
            str(role).strip()
            for role in profile.get("target_roles", [])
            if str(role).strip()
        )
    )
    profile["is_foundational"] = bool(profile.get("is_foundational"))
    if profile["is_foundational"]:
        profile["target_roles"] = []
    elif not profile["target_roles"]:
        raise ValueError("Course classifier returned no target job roles")
    return profile


@lru_cache(maxsize=1)
def _encoder():
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    from sentence_transformers import SentenceTransformer

    try:
        model_path = snapshot_download(EMBED_MODEL, local_files_only=True)
    except LocalEntryNotFoundError:
        model_path = snapshot_download(EMBED_MODEL)
    return SentenceTransformer(model_path)


def _embed(texts: list[str], show_progress: bool = False) -> np.ndarray:
    return _encoder().encode(
        texts,
        batch_size=512,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    ).astype(np.float32)


def _unique_titles(postings_path: Path) -> list[str]:
    titles = set()
    with postings_path.open(encoding="utf-8") as file:
        for line in file:
            posting = json.loads(line)
            title = " ".join(str(posting.get("job_title") or "").casefold().split())
            if title:
                titles.add(title)
    return sorted(titles)


def _title_bank(postings_path: Path) -> tuple[list[str], np.ndarray]:
    """Build the semantic job-title bank once and reuse it for every upload."""
    source = postings_path.stat()
    if TITLE_CACHE.exists():
        cached = np.load(TITLE_CACHE, allow_pickle=False)
        cache_is_current = (
            int(cached["source_size"]) == source.st_size
            and int(cached["source_mtime_ns"]) == source.st_mtime_ns
            and str(cached["model"]) == EMBED_MODEL
        )
        if cache_is_current:
            return cached["titles"].tolist(), cached["vectors"]

    titles = _unique_titles(postings_path)
    print(f"Building title cache: {len(titles):,} unique job titles")
    vectors = _embed(titles, show_progress=True)
    np.savez(
        TITLE_CACHE,
        titles=np.asarray(titles, dtype=np.str_),
        vectors=vectors,
        source_size=np.int64(source.st_size),
        source_mtime_ns=np.int64(source.st_mtime_ns),
        model=np.asarray(EMBED_MODEL),
    )
    return titles, vectors


def _retrieve_titles(
    profile: dict,
    titles: list[str],
    title_vectors: np.ndarray,
    per_query: int = 25,
) -> list[dict]:
    """Semantic retrieval only; the LLM verifier makes the inclusion decision."""
    if per_query < 1:
        raise ValueError("per_query must be at least 1")

    queries = [profile["field_description"], *profile["target_roles"]]
    query_vectors = _embed(queries)
    similarities = title_vectors @ query_vectors.T

    candidate_scores: dict[int, float] = {}
    keep = min(per_query, len(titles))
    for query_index in range(similarities.shape[1]):
        column = similarities[:, query_index]
        indices = np.argpartition(column, -keep)[-keep:]
        for index in indices:
            score = float(column[index])
            candidate_scores[int(index)] = max(
                score, candidate_scores.get(int(index), -1.0)
            )

    ordered = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
    return [
        {"title": titles[index], "semantic_score": round(score, 4)}
        for index, score in ordered
    ]


def _verify_titles(
    profile: dict,
    candidates: list[dict],
    batch_size: int = 80,
) -> dict[str, dict]:
    """Verify candidate titles in batches; posting skills are never shown."""
    import ollama

    verified = {}
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start:start + batch_size]
        batch_number = start // batch_size + 1
        batch_count = (len(candidates) + batch_size - 1) // batch_size
        print(
            f"  verifying title batch {batch_number}/{batch_count} "
            f"with {VERIFY_MODEL}",
            flush=True,
        )
        numbered = "\n".join(
            f"{index}. {candidate['title']}"
            for index, candidate in enumerate(batch)
        )
        response = ollama.chat(
            model=VERIFY_MODEL,
            think=False,
            options={"temperature": 0, "num_ctx": 8192},
            format=_VERIFY_SCHEMA,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Independently verify whether each JOB TITLE directly "
                        "belongs to the supplied labour-market field as its PRIMARY "
                        "job function. Use only the title and field definition. "
                        "Accept seniority and specialization variants of direct "
                        "roles. When a title is ambiguous, reject it unless the field "
                        "identity is explicit in the title; do not infer relevance "
                        "from generic words such as model, training, implementation, "
                        "data, analyst, specialist, developer, or engineer alone. "
                        "Reject adjacent legal, sales, academic, executive, and "
                        "generic roles. Return one decision for every numbered title."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Field: {profile['field_name']}\n"
                        f"Definition: {profile['field_description']}\n"
                        f"Representative roles: {', '.join(profile['target_roles'])}\n\n"
                        f"CANDIDATE JOB TITLES:\n{numbered}"
                    ),
                },
            ],
        )
        decisions = json.loads(response.message.content).get("decisions", [])
        by_index = {
            int(item["index"]): item
            for item in decisions
            if isinstance(item, dict) and isinstance(item.get("index"), int)
        }
        for index, candidate in enumerate(batch):
            decision = by_index.get(index, {})
            if decision.get("is_related") is True:
                verified[candidate["title"]] = {
                    "semantic_score": candidate["semantic_score"],
                    "verification_reason": str(decision.get("reason", "")).strip(),
                }
    return verified


def _select_postings(postings_path: Path, verified_titles: dict[str, dict]) -> list[dict]:
    selected = []
    with postings_path.open(encoding="utf-8") as file:
        for line in file:
            posting = json.loads(line)
            title = " ".join(str(posting.get("job_title") or "").casefold().split())
            evidence = verified_titles.get(title)
            if evidence:
                selected.append({"field_evidence": evidence, **posting})
    return selected


def filter_related_field(
    course_pdf: str | Path,
    postings_path: str | Path = DEFAULT_POSTINGS,
    per_query: int = 15,
) -> dict:
    """Dynamically select the fair job-market cohort for one uploaded syllabus.

    The uploaded PDF determines only the field definition and representative role
    queries. Actual posting inclusion is based on semantic job-title retrieval plus
    independent LLM title verification, never on overlap with course skills.
    """
    pdf_path = Path(course_pdf).expanduser().resolve()
    market_path = Path(postings_path).expanduser().resolve()
    if not market_path.is_file():
        raise FileNotFoundError(f"Job-posting dataset not found: {market_path}")

    document_text = _read_pdf(pdf_path)
    profile = _course_profile(pdf_path, document_text)
    if profile["is_foundational"]:
        return {
            "source_pdf": str(pdf_path),
            "field_profile": profile,
            "selection": {
                "status": "not_market_facing",
                "candidate_titles": 0,
                "verified_titles": 0,
                "postings": 0,
            },
            "postings": [],
        }

    titles, title_vectors = _title_bank(market_path)
    candidates = _retrieve_titles(
        profile, titles, title_vectors, per_query=per_query
    )
    verified = _verify_titles(profile, candidates)
    postings = _select_postings(market_path, verified)
    return {
        "source_pdf": str(pdf_path),
        "field_profile": profile,
        "selection": {
            "status": "field_cohort_ready",
            "candidate_titles": len(candidates),
            "verified_titles": len(verified),
            "postings": len(postings),
        },
        "verified_job_titles": [
            {"title": title, **evidence}
            for title, evidence in sorted(verified.items())
        ],
        "postings": postings,
    }


def _save_result(result: dict) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    source_name = Path(result["source_pdf"]).stem
    path = OUTPUT_DIR / f"{source_name}.field.json"
    path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter job postings to the field of uploaded course PDFs."
    )
    parser.add_argument("course_pdfs", nargs="+", help="One or more course PDF paths")
    parser.add_argument("--postings", default=str(DEFAULT_POSTINGS))
    parser.add_argument("--per-query", type=int, default=15)
    args = parser.parse_args()

    summaries = []
    for course_pdf in args.course_pdfs:
        print(f"\nProcessing: {course_pdf}")
        result = filter_related_field(
            course_pdf,
            postings_path=args.postings,
            per_query=args.per_query,
        )
        output_path = _save_result(result)
        profile = result["field_profile"]
        summary = {
            "course": profile["course_title"],
            "field": profile["field_name"],
            "foundational": profile["is_foundational"],
            **result["selection"],
            "output": str(output_path),
        }
        summaries.append(summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\nAll courses")
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""
KeyBERT baseline benchmark on SkillSpan.

Mirrors skillspan_benchmark.py but replaces the qwen3:14b LLM call with KeyBERT
keyword extraction. Output JSON has the SAME shape as skillspan_benchmark.py so
skill_evaluation_metric.py can score it unchanged.

Usage:
  pip install keybert sentence-transformers
  python keybert_benchmark.py
  python keybert_benchmark.py --top_n 15 --output outputs/skillspan_benchmark/keybert.json
  python skill_evaluation_metric.py --input outputs/skillspan_benchmark/keybert.json
"""

import argparse
import json
import os
from datetime import datetime

from datasets import load_dataset
from keybert import KeyBERT


# ── Reused SkillSpan loading (mirrors skillspan_benchmark.py) ──

def load_skillspan(split="all"):
    print(f"Loading SkillSpan dataset from HuggingFace (split={split})...")
    if split == "all":
        from datasets import concatenate_datasets
        ds_train = load_dataset("jjzha/skillspan", split="train")
        ds_val = load_dataset("jjzha/skillspan", split="validation")
        ds_test = load_dataset("jjzha/skillspan", split="test")
        ds_train = ds_train.map(lambda x: {"idx": f"train_{x['idx']}"})
        ds_val = ds_val.map(lambda x: {"idx": f"val_{x['idx']}"})
        ds_test = ds_test.map(lambda x: {"idx": f"test_{x['idx']}"})
        ds = concatenate_datasets([ds_train, ds_val, ds_test])
    else:
        ds = load_dataset("jjzha/skillspan", split=split)
    print(f"Loaded {len(ds)} rows")
    return ds


def bio_to_skills(tokens, tags):
    skills, current = [], []
    for token, tag in zip(tokens, tags):
        if tag == "B":
            if current:
                skills.append(" ".join(current))
            current = [token]
        elif tag == "I" and current:
            current.append(token)
        else:
            if current:
                skills.append(" ".join(current))
                current = []
    if current:
        skills.append(" ".join(current))
    return skills


def group_by_job(dataset):
    jobs = {}
    for row in dataset:
        idx = row["idx"]
        sentence = " ".join(row["tokens"])
        skills = bio_to_skills(row["tokens"], row["tags_skill"])
        knowledge = bio_to_skills(row["tokens"], row["tags_knowledge"])

        if idx not in jobs:
            jobs[idx] = {"text": "", "human_skills": [], "source": row["source"]}

        jobs[idx]["text"] += sentence + " "
        jobs[idx]["human_skills"].extend(skills)
        jobs[idx]["human_skills"].extend(knowledge)

    for idx in jobs:
        jobs[idx]["text"] = jobs[idx]["text"].strip()
        jobs[idx]["human_skills"] = list(dict.fromkeys(jobs[idx]["human_skills"]))

    return jobs


# ── KeyBERT extraction ──

def extract_keybert_skills(kw_model, text, top_n=15, ngram_range=(1, 3),
                           use_mmr=True, diversity=0.7):
    """Extract top-N keyphrases via KeyBERT, return list of strings."""
    if not text or not text.strip():
        return []
    try:
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=ngram_range,
            stop_words="english",
            top_n=top_n,
            use_mmr=use_mmr,
            diversity=diversity,
        )
        return [phrase for phrase, _score in keywords]
    except Exception as e:
        print(f"  [ERROR] KeyBERT extraction failed: {e}")
        return []


# ── Main ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of jobs")
    parser.add_argument("--model", default="all-MiniLM-L6-v2",
                        help="Sentence-transformer backbone for KeyBERT (default: all-MiniLM-L6-v2)")
    parser.add_argument("--top_n", type=int, default=15,
                        help="Top-N keyphrases per job (matches skillspan 5-15 default)")
    parser.add_argument("--ngram_min", type=int, default=1, help="Minimum n-gram size")
    parser.add_argument("--ngram_max", type=int, default=3, help="Maximum n-gram size")
    parser.add_argument("--diversity", type=float, default=0.7,
                        help="MMR diversity factor (0=more redundant, 1=more diverse)")
    parser.add_argument("--no_mmr", action="store_true",
                        help="Disable MMR diversity (pure top-N cosine)")
    parser.add_argument("--split", default="all",
                        choices=["train", "validation", "test", "all"],
                        help="Dataset split (default: all = 263 jobs)")
    parser.add_argument("--output", default=None,
                        help="Output JSON path. Default: outputs/skillspan_benchmark/<timestamp>/keybert.json")
    args = parser.parse_args()

    # Output path
    if args.output is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(project_root, "outputs", "skillspan_benchmark", timestamp)
        os.makedirs(run_dir, exist_ok=True)
        args.output = os.path.join(run_dir, "keybert.json")
    else:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Output: {args.output}")
    print(f"KeyBERT backbone: {args.model}")
    print(f"top_n={args.top_n}, ngram={args.ngram_min}-{args.ngram_max}, "
          f"use_mmr={not args.no_mmr}, diversity={args.diversity}\n")

    # Load and group SkillSpan
    ds = load_skillspan(split=args.split)
    jobs = group_by_job(ds)
    print(f"Grouped into {len(jobs)} job postings")

    job_list = list(jobs.items())
    if args.limit:
        job_list = job_list[:args.limit]
    print(f"Processing {len(job_list)} jobs with KeyBERT\n")

    # Load KeyBERT once (slow first call, fast after)
    print("Loading KeyBERT model...")
    kw_model = KeyBERT(model=args.model)
    print("Model loaded.\n")

    # Run KeyBERT on each job
    all_results = []
    for i, (idx, job) in enumerate(job_list, 1):
        print(f"[{i}/{len(job_list)}] Job idx={idx} "
              f"(source={job['source']}, human_skills={len(job['human_skills'])})")

        llm_skills = extract_keybert_skills(
            kw_model,
            job["text"],
            top_n=args.top_n,
            ngram_range=(args.ngram_min, args.ngram_max),
            use_mmr=not args.no_mmr,
            diversity=args.diversity,
        )
        print(f"  KeyBERT extracted {len(llm_skills)} skills: "
              f"{llm_skills[:5]}{'...' if len(llm_skills) > 5 else ''}")

        # Note: field name stays "llm_skills" so skill_evaluation_metric.py works unchanged
        result = {
            "llm_skills": llm_skills,
            "human_skills": job["human_skills"],
            "job_title": f"SkillSpan job idx={idx}",
            "source": job["source"],
            "skillspan_idx": idx,
            "method": "keybert",
            "keybert_backbone": args.model,
        }
        all_results.append(result)

        # Incremental save
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Saved {len(all_results)} jobs to {args.output}")
    print(f"\nNext step — score it with:")
    print(f"  python skill_evaluation_metric.py --input {args.output}")


if __name__ == "__main__":
    main()

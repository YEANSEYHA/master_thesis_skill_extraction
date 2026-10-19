
import json
import requests
from datasets import load_dataset
import argparse


# ── TASK 1: Load SkillSpan ──

def load_skillspan(split="test"):
    print(f"Loading SkillSpan dataset from HuggingFace (split={split})...")
    if split == "all":
        from datasets import concatenate_datasets
        ds_train = load_dataset("jjzha/skillspan", split="train")
        ds_val = load_dataset("jjzha/skillspan", split="validation")
        ds_test = load_dataset("jjzha/skillspan", split="test")
        # Prefix idx to avoid overlap across splits
        ds_train = ds_train.map(lambda x: {"idx": f"train_{x['idx']}"})
        ds_val = ds_val.map(lambda x: {"idx": f"val_{x['idx']}"})
        ds_test = ds_test.map(lambda x: {"idx": f"test_{x['idx']}"})
        ds = concatenate_datasets([ds_train, ds_val, ds_test])
    else:
        ds = load_dataset("jjzha/skillspan", split=split)
    print(f"Loaded {len(ds)} rows")
    return ds

def bio_to_skills(token, tags):
    skills = []
    current = []
    for token, tag in zip(token, tags):
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

    # Clean up: strip text and remove duplicate skills
    for idx in jobs:
        jobs[idx]["text"] = jobs[idx]["text"].strip()
        jobs[idx]["human_skills"] = list(dict.fromkeys(jobs[idx]["human_skills"]))

    return jobs

def save_jobs(jobs, output_path="skill_span_jobs.json"):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(jobs)} jobs to {output_path}")


def build_skill_prompt(text, skill_range="5-10"):
    return f"""You are given cleaned text from a job posting webpage. Extract the job posting details into structured JSON.

General Rules:
- Only extract information from the job posting content. Ignore ads, navigation, similar jobs, or unrelated text.
- If a field is not found, set its value to null.
- Do NOT hallucinate or invent information that is not in the text.
- Keep original language and wording where applicable.

Field-Specific Rules:
- "job_title": The exact job title as stated in the posting.
- "job_description": The FULL original text of responsibilities, duties, and role description. Also include any section labeled "Job Responsibility" or "Responsibilities". Combine them if both exist. Do NOT summarize or shorten.
- "job_requirement": The FULL original text of qualifications, experience, education, and requirements. Do NOT summarize or shorten.
- "skills": Extract {skill_range} skills explicitly required by this job, including both technical and interpersonal skills. Follow ALL rules below:
  1. Use short, natural phrases (2-5 words). Can be action-verb style ("communicate with customers", "lead a team") or noun-phrase style ("Python programming", "financial reporting", "machine learning"). Use whatever phrasing best captures the skill as stated in the job text.
  2. Named technology, language, tool, or system MUST appear in the skill phrase. (e.g., text says "Dart" → "Dart programming"; text says "AutoCAD" → "AutoCAD design"). Do NOT replace the specific name with a generic description. If the text lists multiple technologies (e.g., "React / Angular / Node.js"), create a SEPARATE skill for EACH one — do NOT merge them into a single generic phrase like "JavaScript frameworks".
  3. One concept per entry. Never use "and" or "&". BAD: "testing and debugging" → GOOD: "debug software", "software testing".
  4. No parentheses. BAD: "state management (BLoC)" → GOOD: "state management".
  5. Include interpersonal and soft skills when explicitly mentioned (e.g., "communicate with customers", "work in teams", "lead a team", "respond to enquiries", "satisfy customers").
  6. No vague terms, degrees, certifications, years of experience, job titles, or benefits. BAD: "JavaScript frameworks", "database management", "programming languages". GOOD: "React development", "MySQL database", "Python programming".
  7. If two skills overlap, keep the more specific one. No duplicates.
  Return [] if no clear skills found.
- "salary": Copy the salary EXACTLY as it appears in the text, character by character. Do NOT modify, convert, or reformat numbers.
- "location": The job location (city, country, or remote).
- "posted_date": The date the job was posted or published. Extract only the date (e.g., "24-02-2026"), strip any time component.
- "company_name": The name of the hiring company or organization.
- "job_category": The main category this job belongs to (e.g., "IT", "Marketing", "Finance", "Engineering").

Return ONLY a valid JSON object with this exact schema:
{{
  "job_title": "string or null",
  "job_description": "string or null",
  "job_requirement": "string or null",
  "skills": ["string"],
  "salary": "string or null",
  "location": "string or null",
  "posted_date": "string or null",
  "company_name": "string or null",
  "job_category": "string or null"
}}

=== FEW-SHOT EXAMPLE ===

Input text:
"Data Analyst at TechCorp. Responsibilities: Analyze large datasets using SQL and Python. Build dashboards in Tableau and Power BI. Collaborate with stakeholders to define KPIs. Requirements: Proficiency in SQL and Python. Experience with Tableau or Power BI. Strong communication and presentation skills. Knowledge of statistical analysis and A/B testing. Salary: $70,000 - $90,000/year. Location: London, UK. Posted: 15-01-2026."

Correct output:
{{
  "job_title": "Data Analyst",
  "job_description": "Analyze large datasets using SQL and Python. Build dashboards in Tableau and Power BI. Collaborate with stakeholders to define KPIs.",
  "job_requirement": "Proficiency in SQL and Python. Experience with Tableau or Power BI. Strong communication and presentation skills. Knowledge of statistical analysis and A/B testing.",
  "skills": ["SQL querying", "Python programming", "Tableau visualization", "Power BI dashboards", "statistical analysis", "A/B testing", "stakeholder communication"],
  "salary": "$70,000 - $90,000/year",
  "location": "London, UK",
  "posted_date": "15-01-2026",
  "company_name": "TechCorp",
  "job_category": "IT"
}}

Notice in the example above:
- Each skill is 2-3 words, one concept only
- "SQL and Python" was split into "SQL querying" and "Python programming" (never use "and")
- "Tableau or Power BI" became two separate skills
- "communication and presentation skills" became "stakeholder communication" (specific, no "and")
- No generic filler skills like "teamwork" or "problem solving"

=== FEW-SHOT EXAMPLE 2 ===

Input text:
"Marketing Executive at BrightWave Agency. Job Responsibilities: Plan and execute digital marketing campaigns across Facebook Ads and Google Ads. Manage social media accounts and create engaging content using Canva. Analyze campaign performance with Google Analytics and prepare monthly reports in Excel. Coordinate with clients to understand brand objectives and deliver presentations. Requirements: Experience with Facebook Ads and Google Ads campaign management. Proficiency in Canva for content creation. Familiarity with Google Analytics. Strong skills in Excel for reporting. Excellent client communication and presentation abilities. Ability to manage multiple projects under tight deadlines. Salary: $3,500 - $4,500/month. Location: Phnom Penh, Cambodia. Posted: 10-02-2026."

Correct output:
{{
  "job_title": "Marketing Executive",
  "job_description": "Plan and execute digital marketing campaigns across Facebook Ads and Google Ads. Manage social media accounts and create engaging content using Canva. Analyze campaign performance with Google Analytics and prepare monthly reports in Excel. Coordinate with clients to understand brand objectives and deliver presentations.",
  "job_requirement": "Experience with Facebook Ads and Google Ads campaign management. Proficiency in Canva for content creation. Familiarity with Google Analytics. Strong skills in Excel for reporting. Excellent client communication and presentation abilities. Ability to manage multiple projects under tight deadlines.",
  "skills": ["Facebook Ads campaigns", "Google Ads campaigns", "Canva content creation", "Google Analytics reporting", "Excel reporting", "client communication", "deliver presentations", "manage multiple projects"],
  "salary": "$3,500 - $4,500/month",
  "location": "Phnom Penh, Cambodia",
  "posted_date": "10-02-2026",
  "company_name": "BrightWave Agency",
  "job_category": "Marketing"
}}

Notice in this example:
- "Facebook Ads and Google Ads" split into two separate skills with the tool name kept
- "client communication and presentation abilities" split into "client communication" and "deliver presentations"
- "manage multiple projects" kept as interpersonal skill (explicitly mentioned)
- No generic terms like "digital marketing" when specific tools are named
- BAD: "social media management" (too vague) → GOOD: specific tool names kept

=== NOW EXTRACT FROM THIS TEXT ===

Text:
{text}""".strip()


def ask_llm(prompt, model_name="qwen3:14b"):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "system": "You are a JSON data extraction tool. Output only valid JSON.",
                "stream": False,
                "format": "json",
                "options": {"temperature": 0, "num_ctx": 32768},
                "think": False,
            },
            timeout=1200,
        )
        response.raise_for_status()
        raw = response.json().get("response", "")
        try:
            parsed = json.loads(raw)
            # LLM returns full 9-field JSON, we only need skills
            return parsed.get("skills", [])
        except json.JSONDecodeError:
            print(f"  [ERROR] Failed to parse JSON: {raw[:100]}")
            return []
    except requests.exceptions.ReadTimeout:
        print(f"  [TIMEOUT] Skipping — exceeded 1200s")
        return []
    except requests.exceptions.ConnectionError:
        print(f"  [CONNECTION ERROR] Skipping — could not reach Ollama")
        return []


def main():
    import os
    from datetime import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of jobs")
    parser.add_argument("--model", default="llama3.1:8b", help="Ollama model name")
    parser.add_argument("--output", default=None, help="Output file path (default: outputs/skillspan_benchmark/<timestamp>/skillspan_benchmark.json)")
    parser.add_argument("--split", default="all", choices=["train", "validation", "test", "all"],
                        help="Dataset split to use (default: all for full 263 jobs)")
    parser.add_argument("--skills", default="5-10",
                        help="Skill extraction range (e.g., '5-10', '8-12', '10-15'). Default: 5-10")
    parser.add_argument("--resume", default=None,
                        help="Path to existing output JSON to resume from (skips already-processed jobs)")
    args = parser.parse_args()

    # Set up output path: outputs/skillspan_benchmark/<timestamp>/skillspan_benchmark.json
    if args.output is None:
        project_root = os.path.dirname(os.path.abspath(__file__))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(project_root, "outputs", "skillspan_benchmark", timestamp)
        os.makedirs(run_dir, exist_ok=True)
        args.output = os.path.join(run_dir, "skillspan_benchmark.json")
    else:
        # Ensure output directory exists for custom paths
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Output: {args.output}\n")

    # Step 1-3: Load SkillSpan → group by job → get human skills
    ds = load_skillspan(split=args.split)
    jobs = group_by_job(ds)
    save_jobs(jobs)
    print(f"Grouped into {len(jobs)} job postings")

    job_list = list(jobs.items())
    if args.limit:
        job_list = job_list[:args.limit]
    print(f"Processing {len(job_list)} jobs with model={args.model}, skills={args.skills}\n")

    # Step 4: Run LLM on each job (with optional resume)
    all_results = []
    skip_idxs = set()
    if args.resume:
        with open(args.resume, "r", encoding="utf-8") as f:
            all_results = json.load(f)
        skip_idxs = {r["skillspan_idx"] for r in all_results}
        args.output = args.resume  # overwrite same file
        print(f"Resuming: loaded {len(all_results)} existing results, skipping those jobs\n")

    for i, (idx, job) in enumerate(job_list, 1):
        if idx in skip_idxs:
            continue
        print(f"[{i}/{len(job_list)}] Job idx={idx} (source={job['source']}, human_skills={len(job['human_skills'])})")

        prompt = build_skill_prompt(job["text"], skill_range=args.skills)
        llm_skills = ask_llm(prompt, model_name=args.model)
        print(f"  LLM extracted {len(llm_skills)} skills: {llm_skills[:5]}{'...' if len(llm_skills) > 5 else ''}")

        result = {
            "llm_skills": llm_skills,
            "human_skills": job["human_skills"],
            "job_title": f"SkillSpan job idx={idx}",
            "source": job["source"],
            "skillspan_idx": idx,
        }
        all_results.append(result)

        # Save after each job (incremental)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Saved {len(all_results)} jobs to {args.output}")



if __name__ == "__main__":
    main()


import sys
import os
import json
import requests
import trafilatura
from bs4 import BeautifulSoup

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
from utils import read_all_htmls

from rich.console import Console

console = Console()

def build_prompt(text_content: str) -> str:
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
- "skills": Extract 5-10 skills explicitly required by this job, including both technical and interpersonal skills. Follow ALL rules below:
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
"Marketing Executive at BrightWave Agency. Job Responsibilities: Plan and execute digital marketing campaigns across Facebook Ads and Google Ads. Manage social media accounts and create engaging content using Canva. Analyze campaign performance with Google Analytics and prepare monthly reports in Excel. Coordinate with clients to understand brand objectives and deliver presentations. Requirements: Experience with Facebook Ads and Google Ads campaign management. Proficiency in Canva for content creation. Familiarity with Google Analytics. Strong skills in Excel for reporting. Excellent client communication and presentation abilities. Ability to manage multiple projects under tight deadlines. Salary: $800 - $1,200/month. Location: Phnom Penh, Cambodia. Posted: 10-02-2026."

Correct output:
{{
  "job_title": "Marketing Executive",
  "job_description": "Plan and execute digital marketing campaigns across Facebook Ads and Google Ads. Manage social media accounts and create engaging content using Canva. Analyze campaign performance with Google Analytics and prepare monthly reports in Excel. Coordinate with clients to understand brand objectives and deliver presentations.",
  "job_requirement": "Experience with Facebook Ads and Google Ads campaign management. Proficiency in Canva for content creation. Familiarity with Google Analytics. Strong skills in Excel for reporting. Excellent client communication and presentation abilities. Ability to manage multiple projects under tight deadlines.",
  "skills": ["Facebook Ads campaigns", "Google Ads campaigns", "Canva content creation", "Google Analytics reporting", "Excel reporting", "client communication", "deliver presentations", "manage multiple projects"],
  "salary": "$800 - $1,200/month",
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
{text_content}""".strip()

def saveJson(jsonData, outputDir):
    with open(outputDir, "w", encoding="utf-8") as f:
        json.dump(jsonData, f, ensure_ascii=False, indent=2)


def extract_url_from_html(html: str) -> str | None:
    """Extract the canonical URL from HTML, trying multiple strategies."""
    import urllib.parse
    soup = BeautifulSoup(html, "html.parser")

    # 1. og:url meta tag
    og_url = soup.find("meta", property="og:url")
    if og_url and og_url.get("content"):
        return og_url["content"].strip()

    # 2. canonical link tag
    canonical = soup.find("link", rel="canonical")
    if canonical and canonical.get("href"):
        return canonical["href"].strip()

    # 3. data-share attribute (relative URL like "/jobs/jobs-show-22045-.htm")
    #    Reconstruct using base domain from any absolute href on the page
    data_share = soup.find(attrs={"data-share": True})
    if data_share:
        share_path = data_share["data-share"].strip()
        # Find any absolute href to extract the base domain
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("http"):
                parsed = urllib.parse.urlparse(href)
                base = f"{parsed.scheme}://{parsed.netloc}"
                return base + share_path
        # No absolute href found, return path as-is
        if share_path.startswith("http"):
            return share_path
        return share_path

    # 4. fromurl parameter in login/redirect links
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "fromurl=" in href:
            try:
                qs = urllib.parse.urlparse(href).query
                params = urllib.parse.parse_qs(qs)
                if "fromurl" in params:
                    return urllib.parse.unquote(params["fromurl"][0])
            except Exception:
                pass

    return None


def clean_html_trafilatura(html: str, max_chars: int = 40000) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # ── 1. Extract structured metadata before anything gets stripped ──
    meta_lines = []

    # Job title
    job_title_el = (
        soup.find(["h1", "h2", "h3"], class_=lambda c: c and any(k in c for k in ["job", "title", "position"]))
        or soup.find("h1")
        or soup.find("h2")
    )
    if job_title_el:
        t = job_title_el.get_text(" ", strip=True)
        if t:
            meta_lines.append(f"Job Title: {t}")

    # Company name
    company_el = soup.find(["a", "span", "div", "h3", "h4"],
                           class_=lambda c: c and any(k in c for k in ["company", "employer", "org"]))
    if company_el:
        t = company_el.get_text(" ", strip=True)
        if t and len(t) < 100:
            meta_lines.append(f"Company: {t}")

    # Salary
    salary_el = soup.find(["span", "div", "li", "h4"],
                          class_=lambda c: c and any(k in c for k in ["salary", "compensation", "wage", "pay"]))
    if salary_el:
        t = salary_el.get_text(" ", strip=True)
        if t:
            meta_lines.append(f"Salary: {t}")

    # Location
    location_el = soup.find(["span", "div", "li"],
                            class_=lambda c: c and any(k in c for k in ["location", "address", "region"]))
    if location_el:
        t = location_el.get_text(" ", strip=True)
        if t and len(t) < 150:
            meta_lines.append(f"Location: {t}")

    # ── 2. Remove noise elements before get_text() ──
    noise_tags = ["script", "style", "noscript", "nav", "footer", "header", "iframe", "svg"]
    for tag in soup(noise_tags):
        tag.decompose()

    # Remove common non-job sections by class/id keywords
    noise_keywords = ["cookie", "consent", "similar", "recommend", "related",
                      "newsletter", "subscribe", "social-share", "breadcrumb",
                      "advertisement", "ad-", "popup", "modal", "banner"]
    for el in list(soup.find_all(True, class_=True)):
        if not el.attrs or not hasattr(el, "name"):
            continue
        cls = el.get("class")
        if not cls:
            continue
        classes = " ".join(cls).lower()
        if any(k in classes for k in noise_keywords):
            el.decompose()
    for el in list(soup.find_all(True, id=True)):
        if not el.attrs or not hasattr(el, "name"):
            continue
        el_id = el.get("id")
        if not el_id:
            continue
        if any(k in el_id.lower() for k in noise_keywords):
            el.decompose()

    # ── 3. Main content via trafilatura ──
    main_text = trafilatura.extract(
        html,
        include_tables=True,
        include_links=True,
        favor_recall=True
    ) or ""

    # ── 4. Body text (noise already removed in step 2, safe to always include) ──
    body_text = soup.get_text(separator="\n", strip=True)

    # ── 5. Combine: metadata + main content + body text ──
    combined = "\n".join(meta_lines) + "\n\n" + main_text + "\n\n" + body_text

    # ── 6. Deduplicate lines and filter short junk ──
    seen = set()
    result = []
    for line in combined.splitlines():
        stripped = line.strip()
        if not stripped or stripped in seen:
            continue
        # Skip lines that look like navigation/menu junk (very short, no spaces)
        if len(stripped) < 3:
            continue
        seen.add(stripped)
        result.append(stripped)

    cleaned = "\n".join(result)

    # ── 7. Truncate if too long for LLM context ──
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars] + "\n\n[... truncated]"

    return cleaned

def ask_llm(model_name: str, prompt: str, base_url: str = "http://localhost:11434", timeout: int = 300) -> str:
    response = requests.post(
        f"{base_url}/api/generate",
        json={
            "model": model_name,
            "prompt": prompt,
            "system": "You are a JSON data extraction tool. Output only valid JSON.",
            "stream": False,
            "format": "json",
            "options": {"temperature":0,"num_ctx":32768},
            "think": False
        },
        timeout=timeout
    )
    response.raise_for_status()
    return response.json().get("response","")

def main(input_dir: str | None = None, limit: int | None = None):
    import time
    from datetime import datetime

    _input = input_dir or "../data_collection/data/job_card"
    htmls = read_all_htmls(_input, limit=limit or 200)
    total = len(htmls)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(project_root, "outputs", "stage2", timestamp)
    os.makedirs(run_dir, exist_ok=True)
    output_filename = os.path.join(run_dir, "stage2_result.json")

    console.print(f"[bold]Output: {output_filename}[/bold]\n")

    all_results = []
    errors = 0
    start_time = time.time()

    total = len(htmls)

    for idx, item in enumerate(htmls, 1):
        filename =item["filename"]
        raw_html = item["content"]

        cleaned_text = clean_html_trafilatura(raw_html)
        prompt = build_prompt(cleaned_text)

        # ── INPUT: show what we feed to the LLM ──
        # console.print(f"\n{'═' * 70}")
        # console.print(f"[bold cyan][{idx}/{total}][/bold cyan] [dim]{filename}[/dim]")
        # console.print(f"[bold yellow]── INPUT (fed to LLM) ({len(cleaned_text):,} chars) ──[/bold yellow]")
        # console.print(f"[dim]{cleaned_text}[/dim]")

        try:
            job_start = time.time()
            response = ask_llm("llama3.1:8b", prompt)
            job_duration = time.time() - job_start

            parsed = json.loads(response)
            parsed["source_file"] = filename
            parsed["full_url"] = extract_url_from_html(raw_html)
            all_results.append({
                "input": cleaned_text,
                "output": parsed,
            })

            # Save after each job (incremental — prevents data loss)
            saveJson(all_results, output_filename)

            # Progress info
            elapsed = time.time() - start_time
            avg_per_job = elapsed / idx
            remaining = avg_per_job * (total - idx)
            mins_remaining = remaining / 60

            # ── OUTPUT: show what the LLM returned ──
            out = parsed
            title = (out.get("job_title") or "Unknown")[:50]
            skills_count = len(out.get("skills", []))
            skills_preview = ", ".join(f'"{s}"' for s in out.get("skills", []))

            console.print(f"[bold green]── OUTPUT (LLM result) ──[/bold green]")
            console.print(f"  [green]✓[/green] [bold]{title}[/bold] | {skills_count} skills | {job_duration:.1f}s | ETA: {mins_remaining:.1f}min")
            console.print(f"  [dim]company:[/dim]  {out.get('company_name') or '—'}")
            console.print(f"  [dim]location:[/dim] {out.get('location') or '—'}")
            console.print(f"  [dim]salary:[/dim]   {out.get('salary') or '—'}")
            console.print(f"  [dim]category:[/dim] {out.get('job_category') or '—'}")
            console.print(f"  [dim]posted:[/dim]   {out.get('posted_date') or '—'}")
            console.print(f"  [dim]url:[/dim]      {out.get('full_url') or '—'}")
            console.print(f"  [dim]skills:[/dim]   [cyan]{skills_preview}[/cyan]")

        except Exception as e:
            errors += 1
            console.print(f"[bold red]── OUTPUT (ERROR) ──[/bold red]")
            console.print(f"  [red]✗[/red] {filename} — {e}")

    # Final summary
    total_time = time.time() - start_time
    console.print(f"\n[bold green]Done![/bold green] {len(all_results)}/{total} processed in {total_time/60:.1f}min ({errors} errors)")
        

if __name__ == "__main__":
    main()
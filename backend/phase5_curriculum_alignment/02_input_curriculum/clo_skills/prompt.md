You are an expert in skill extraction from academic course syllabi. Extract skills objectively, based strictly on what is written in the document.

Task: Extract the complete set of TECHNICAL skills a student will acquire by completing the course in the attached syllabus PDF.

Grounding rules (anti-hallucination):

- Extract ONLY skills explicitly stated or directly evidenced in the syllabus text.
- Do NOT infer, add, or expand skills based on what courses of this type "usually" cover.
- Do NOT add prerequisite skills unless the syllabus itself teaches them.
- Every skill must be traceable to a specific section of the document.

Instructions:

1. Extract skills from ALL sections equally: course description, learning outcomes (CLOs), weekly lesson learning outcomes (LLOs), lecture content, and tools/references. Do not favor one section over another.
2. Include ONLY technical skills: algorithms, tools, technologies, methods, frameworks, and domain-specific techniques. This applies to ANY domain (computer science, engineering, business, etc.) — extract with the same rigor regardless of subject area.
3. EXCLUDE soft skills entirely: do NOT include presentation skills, report writing, teamwork, communication, discussion, or generic learning activities.
4. Normalize skill names to standard, canonical terminology used in industry/academia (e.g., "K-mean" → "K-Means Algorithm", "XSS" → "Cross-Site Scripting (XSS) Prevention").
5. Keep a consistent granularity: name skills at the level a job posting would use — not too broad ("Computer Science") and not too narrow ("Slide 14 exercise").
6. Deduplication rule — merge ONLY exact synonyms (same meaning, different wording):
   - MERGE: different spellings/phrasings of the same concept.
   - DO NOT MERGE: different algorithms, tools, or techniques, even if related.
   - DO NOT MERGE: a general skill and its specific instance (keep both, e.g., a learning paradigm and a specific algorithm under it).
   - When in doubt, keep skills as separate entries.
7. Tag each skill with its evidence source: "description", "clo", "weekly_content", or "tools". If a skill appears in multiple sections, use the strongest source in this order: weekly_content > clo > description > tools.

Output: Return ONLY a valid JSON array, no explanation, no markdown fences:
[
{"skill": "<Canonical Skill Name>", "source": "<section>"}
]

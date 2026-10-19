Input:

- Course syllabus PDF
- Labor-market skills

Step 1
LLM Skill Extraction

PDF
→ Course Skills

Job Postings
→ Market Skills

Step 2
Semantic Skill Matching

Course Skills
↔
Market Skills

Step 3
Importance-Weighted Coverage

# Importance(skill)

Demand(skill) × Specificity(skill)

# Coverage

Σ Importance(matched)
/
Σ Importance(all)

Step 4
Recommendation

Coverage ≥ 0.70
→ MAINTAIN

0.40 ≤ Coverage < 0.70
→ REVISE

Coverage < 0.40
→ UPDATE

Step 5
Explainability

Return:

- Coverage score
- Recommendation
- Strengths
- Critical gaps
- Suggested additions

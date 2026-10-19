"""Lexical baseline — the unmodified Ahadi (EDM 2022) exact-match approach.

An ablation of the CLO-Skill Alignment method: identical pipeline (CLOs,
job-skill inventory, frequency-weighted coverage, MAINTAIN/REVISE/UPDATE
thresholds) but the matching step uses **word-set overlap on surface forms**
instead of embedding cosine similarity — and **no ESCO normalization**.

Holding everything else constant isolates the contribution of the semantic
+ ESCO components in clo_skill_alignment/. See README.md.
"""

# Phase 6 gap analysis — 20260727T104658Z-475eb3

**V8 tab · cambodia · 9 saved courses · Y3**

| | |
|---|---|
| ran | 2026-07-27T10:46:58+00:00 · 14.5 s |
| code | git 32fab9b4 |
| config hash | `475eb38e6ff0a1eaabd0f377e56dcf53ca500fe19beb84aca06a771af0a90eb6` |
| input hash | `f051a7b241a1bd0445f45720a69938109f301fa44f80495092b14c412794ed3e` |
| cohort | Y3 |

## Inputs

**Market** — `cambodia` · `stage2_result.json` · 254 jobs in segment · segment mode `category`

**Curriculum** — 9 courses (db) · 218 CLO skills · cohort **Y3**

| course | file | CLO skills |
|---|---|---|
| Information System Analysis And Design | `1_information_system_analysis_and_design.json` | 20 |
| Algorithm And Programming | `2_algorithm_and_programming.json` | 17 |
| Combinational And Sequential Logics | `3_combinational_and_sequential_logics.json` | 26 |
| Introduction To Computer Systems And Networks | `4_introduction_to_computer_systems_and_networks.json` | 27 |
| Soft Skills | `5_soft_skills.json` | 2 |
| Object Oriented Programming | `6_object_oriented_programming.json` | 20 |
| Discrete Mathematics | `7_discrete_mathematics.json` | 38 |
| Electronics | `8_electronics.json` | 30 |
| Research Methodology | `9_research_methodology.json` | 38 |

Host verification computed inline (`BAAI/bge-reranker-v2-m3`)

## Parameters

| param | value |
|---|---|
| encoder | `BAAI/bge-base-en-v1.5` |
| covered_at | `0.73` |
| borderline_at | `0.69` |
| host_min | `0.66` |
| top_n | `30` |
| job_source | `cambodia` |
| segment_mode | `category` |
| min_skill_freq | `2` |
| skill_merge_threshold | `0.8` |
| max_skills_to_normalize | `8000` |
| eps | `1e-06` |
| cache_version | `v3` |
| cum_frac | `0.8` |
| spill_min | `0.05` |
| reranker | `BAAI/bge-reranker-v2-m3` |
| rerank_top_k | `3` |
| rerank_supported_at | `0.5` |
| rerank_weak_at | `0.1` |

## Results

KL divergence **0.381** · covered **12 / 30**

### Top skills by demand

| # | skill | jobs | coverage | status | best CLO match |
|---:|---|---:|---:|---|---|
| 1 | technical documentation | 49 | 0.78 | ✓ covered | Information System Analysis And Design — Data Integrity and Data Quality Documentation |
| 2 | database management | 42 | 0.91 | ✓ covered | Information System Analysis And Design — Database Management Systems (DBMS) |
| 3 | java programming | 41 | 0.87 | ✓ covered | Object Oriented Programming — Java |
| 4 | project management | 37 | 0.65 | ✗ gap | Information System Analysis And Design — Database Management Systems (DBMS) |
| 5 | network troubleshooting | 35 | 1.00 | ✓ covered | Introduction To Computer Systems And Networks — Network Troubleshooting |
| 6 | system maintenance | 34 | 0.82 | ✓ covered | Object Oriented Programming — Software Application Maintenance |
| 7 | mysql database | 30 | 0.71 | ◑ borderline | Information System Analysis And Design — Database Management Systems (DBMS) |
| 8 | restful api development | 28 | 0.65 | ✗ gap | Object Oriented Programming — Object-Oriented Analysis and Design |
| 9 | postgresql database | 27 | 0.66 | ✗ gap | Information System Analysis And Design — Relational Database Design |
| 10 | technical support | 27 | 0.69 | ✗ gap | Introduction To Computer Systems And Networks — Network Troubleshooting |
| 11 | ci/cd pipelines | 26 | 0.63 | ✗ gap | Information System Analysis And Design — Information System Components |
| 12 | docker containerization | 24 | 0.61 | ✗ gap | Object Oriented Programming — Encapsulation |
| 13 | git version control | 24 | 0.65 | ✗ gap | Object Oriented Programming — Control Statements |
| 14 | firewall configuration | 23 | 1.00 | ✓ covered | Introduction To Computer Systems And Networks — Firewall Configuration |
| 15 | security compliance | 23 | 0.72 | ◑ borderline | Information System Analysis And Design — Integrity Constraints |
| 16 | server management | 21 | 0.76 | ✓ covered | Introduction To Computer Systems And Networks — Network Device Management |
| 17 | aws cloud platforms | 19 | 0.61 | ✗ gap | Introduction To Computer Systems And Networks — Network Architectures |
| 18 | incident response | 19 | 0.65 | ✗ gap | Object Oriented Programming — Exception Handling |
| 19 | security monitoring | 19 | 0.70 | ◑ borderline | Introduction To Computer Systems And Networks — Packet Tracer |
| 20 | api integration | 18 | 0.67 | ✗ gap | Object Oriented Programming — Interfaces |
| 21 | code reviews | 18 | 0.68 | ✗ gap | Research Methodology — Literature Review |
| 22 | data analysis | 18 | 1.00 | ✓ covered | Research Methodology — Data Analysis |
| 23 | it infrastructure management | 18 | 0.76 | ✓ covered | Introduction To Computer Systems And Networks — Network Infrastructure Design |
| 24 | microsoft office proficiency | 18 | 0.64 | ✗ gap | Information System Analysis And Design — Information System Components |
| 25 | software development | 18 | 0.74 | ✓ covered | Object Oriented Programming — Software Requirements Analysis |
| 26 | system integration | 18 | 0.76 | ✓ covered | Information System Analysis And Design — Information System Components |
| 27 | vmware virtualization | 18 | 0.72 | ◑ borderline | Introduction To Computer Systems And Networks — Virtual Network Design |
| 28 | performance tuning | 17 | 0.70 | ◑ borderline | Object Oriented Programming — Scalable Software Design |
| 29 | disaster recovery planning | 16 | 0.65 | ✗ gap | Research Methodology — Analysis Plans |
| 30 | network administration | 16 | 0.84 | ✓ covered | Introduction To Computer Systems And Networks — Network Device Management |

### Optimized skill plan

Greedy facility-location, stop at 80% of achievable gain → **add 12 of 18** gaps. KL 0.381 → 0.3432

| | skill | jobs | closes | KL after | covered after | action |
|---|---|---:|---:|---:|---:|---|
| **ADD** | project management | 37 | 2.4% | 0.3768 | 13 | Add new project management content |
| **ADD** | security monitoring | 19 | 2.0% | 0.3758 | 15 | Add new security monitoring content |
| **ADD** | microsoft office proficiency | 18 | 1.8% | 0.374 | 16 | Add new microsoft office proficiency content |
| **ADD** | technical support | 27 | 1.8% | 0.3729 | 17 | Extend Object Oriented Programming with explicit technical support content |
| **ADD** | restful api development | 28 | 1.7% | 0.3693 | 19 | Add new restful api development content |
| **ADD** | mysql database | 30 | 1.5% | 0.365 | 21 | Add mysql database content building on Information System Analysis And Design |
| **ADD** | ci/cd pipelines | 26 | 1.3% | 0.3619 | 22 | Add new ci/cd pipelines content |
| **ADD** | docker containerization | 24 | 1.1% | 0.3586 | 23 | Extend Object Oriented Programming with explicit docker containerization content |
| **ADD** | aws cloud platforms | 19 | 1.1% | 0.3562 | 24 | Add new aws cloud platforms content |
| **ADD** | git version control | 24 | 1.0% | 0.3534 | 25 | Add new git version control content |
| **ADD** | performance tuning | 17 | 0.9% | 0.3524 | 26 | Add new performance tuning content |
| **ADD** | security compliance | 23 | 0.9% | 0.3513 | 26 | Add new security compliance content |
| skip | code reviews | 18 | 0.8% | 0.3495 | 27 | Add new code reviews content |
| skip | incident response | 19 | 0.8% | 0.3476 | 28 | Extend Object Oriented Programming with explicit incident response content |
| skip | api integration | 18 | 0.7% | 0.3476 | 28 | Add new api integration content |
| skip | disaster recovery planning | 16 | 0.7% | 0.3463 | 29 | Add new disaster recovery planning content |
| skip | postgresql database | 27 | 0.7% | 0.3446 | 29 | Add new postgresql database content |
| skip | vmware virtualization | 18 | 0.6% | 0.3432 | 30 | Add new vmware virtualization content |

Not selected, and why:

- **code reviews** — only closes 0.8% of the gap (coverage after the plan 0.68)
- **incident response** — only closes 0.8% of the gap (coverage after the plan 0.65)
- **api integration** — absorbed by restful api development (coverage after the plan 0.78)
- **disaster recovery planning** — only closes 0.7% of the gap (coverage after the plan 0.65)
- **postgresql database** — absorbed by mysql database (coverage after the plan 0.80)
- **vmware virtualization** — only closes 0.6% of the gap (coverage after the plan 0.72)

### Gaps with no verified host

No existing CLO shows verified evidence of teaching these — they need new content rather than an extension.

- **project management** — 37 jobs, coverage 0.65 (nearest: Information System Analysis And Design)
- **restful api development** — 28 jobs, coverage 0.65 (nearest: Object Oriented Programming)
- **ci/cd pipelines** — 26 jobs, coverage 0.63 (nearest: Information System Analysis And Design)
- **postgresql database** — 27 jobs, coverage 0.66 (nearest: Information System Analysis And Design)
- **mysql database** — 30 jobs, coverage 0.71 (nearest: Information System Analysis And Design)
- **git version control** — 24 jobs, coverage 0.65 (nearest: Object Oriented Programming)
- **aws cloud platforms** — 19 jobs, coverage 0.61 (nearest: Introduction To Computer Systems And Networks)
- **security compliance** — 23 jobs, coverage 0.72 (nearest: Information System Analysis And Design)
- **microsoft office proficiency** — 18 jobs, coverage 0.64 (nearest: Information System Analysis And Design)
- **api integration** — 18 jobs, coverage 0.67 (nearest: Object Oriented Programming)

---

Generated by `gen_report_data.py`. `report.json` in this directory holds the full machine-readable report.

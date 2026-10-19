# Phase 6 gap analysis — 20260727T104822Z-475eb3

**V8 tab · cambodia · 8 saved courses · Y5**

| | |
|---|---|
| ran | 2026-07-27T10:48:22+00:00 · 13.9 s |
| code | git 32fab9b4 |
| config hash | `475eb38e6ff0a1eaabd0f377e56dcf53ca500fe19beb84aca06a771af0a90eb6` |
| input hash | `917674e944c2ea69a2c8efea6109c400165a71262f8a2c889fe52de7b7073524` |
| cohort | Y5 |

## Inputs

**Market** — `cambodia` · `stage2_result.json` · 254 jobs in segment · segment mode `category`

**Curriculum** — 8 courses (db) · 226 CLO skills · cohort **Y5**

| course | file | CLO skills |
|---|---|---|
| Cloud Computing | `1_cloud_computing.json` | 18 |
| Introduction To Artificial Intelligence | `2_introduction_to_artificial_intelligence.json` | 40 |
| Natural Language Processing | `3_natural_language_processing.json` | 19 |
| Data Mining | `4_data_mining.json` | 55 |
| Image Processing | `5_image_processing.json` | 41 |
| Information Security | `6_information_security.json` | 14 |
| Network Security | `7_network_security.json` | 21 |
| It Project Management | `8_it_project_management.json` | 18 |

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

KL divergence **0.3768** · covered **17 / 30**

### Top skills by demand

| # | skill | jobs | coverage | status | best CLO match |
|---:|---|---:|---:|---|---|
| 1 | technical documentation | 49 | 0.67 | ✗ gap | It Project Management — Project Status Reporting |
| 2 | database management | 42 | 0.91 | ✓ covered | Data Mining — Database Management Systems (DBMS) |
| 3 | java programming | 41 | 0.74 | ✓ covered | Image Processing — Python |
| 4 | project management | 37 | 0.87 | ✓ covered | It Project Management — Project Team Management |
| 5 | network troubleshooting | 35 | 0.77 | ✓ covered | Network Security — Network Devices |
| 6 | system maintenance | 34 | 0.71 | ◑ borderline | Information Security — Security Solutions for System Support and Operations |
| 7 | mysql database | 30 | 0.78 | ✓ covered | Data Mining — Database Queries |
| 8 | restful api development | 28 | 0.68 | ✗ gap | Cloud Computing — Service-Oriented Architecture |
| 9 | postgresql database | 27 | 0.72 | ◑ borderline | Data Mining — Database Queries |
| 10 | technical support | 27 | 0.68 | ✗ gap | Information Security — Security Solutions for System Support and Operations |
| 11 | ci/cd pipelines | 26 | 0.69 | ◑ borderline | Introduction To Artificial Intelligence — Machine Learning Pipeline |
| 12 | docker containerization | 24 | 0.89 | ✓ covered | Cloud Computing — Containerization |
| 13 | git version control | 24 | 0.64 | ✗ gap | It Project Management — Change Management |
| 14 | firewall configuration | 23 | 0.83 | ✓ covered | Network Security — Firewall Technology |
| 15 | security compliance | 23 | 1.00 | ✓ covered | Information Security — Security Compliance |
| 16 | server management | 21 | 0.76 | ✓ covered | Cloud Computing — Load Balancing |
| 17 | aws cloud platforms | 19 | 0.82 | ✓ covered | Cloud Computing — AWS |
| 18 | incident response | 19 | 0.66 | ✗ gap | Network Security — Disaster Recovery Planning |
| 19 | security monitoring | 19 | 0.83 | ✓ covered | Information Security — Information Security Threat Trend Monitoring |
| 20 | api integration | 18 | 0.65 | ✗ gap | Cloud Computing — Platform as a Service (PaaS) |
| 21 | code reviews | 18 | 0.67 | ✗ gap | Image Processing — Huffman Coding |
| 22 | data analysis | 18 | 0.79 | ✓ covered | Data Mining — Correlation Analysis |
| 23 | it infrastructure management | 18 | 0.75 | ✓ covered | Cloud Computing — Infrastructure as a Service (IaaS) |
| 24 | microsoft office proficiency | 18 | 0.61 | ✗ gap | Data Mining — Data Quality Assessment |
| 25 | software development | 18 | 0.71 | ◑ borderline | Cloud Computing — Software as a Service (SaaS) |
| 26 | system integration | 18 | 0.71 | ◑ borderline | Information Security — Security Solutions for System Support and Operations |
| 27 | vmware virtualization | 18 | 0.89 | ✓ covered | Cloud Computing — Virtualization |
| 28 | performance tuning | 17 | 0.74 | ✓ covered | Introduction To Artificial Intelligence — Performance Evaluation and Overfitting/Underfitting Diagnosis |
| 29 | disaster recovery planning | 16 | 1.00 | ✓ covered | Network Security — Disaster Recovery Planning |
| 30 | network administration | 16 | 0.75 | ✓ covered | Network Security — Network Devices |

### Optimized skill plan

Greedy facility-location, stop at 80% of achievable gain → **add 9 of 13** gaps. KL 0.3768 → 0.3478

| | skill | jobs | closes | KL after | covered after | action |
|---|---|---:|---:|---:|---:|---|
| **ADD** | technical documentation | 49 | 2.8% | 0.3706 | 19 | Add new technical documentation content |
| **ADD** | software development | 18 | 2.2% | 0.3684 | 20 | Add software development content building on Cloud Computing |
| **ADD** | system maintenance | 34 | 2.2% | 0.3661 | 22 | Add new system maintenance content |
| **ADD** | microsoft office proficiency | 18 | 1.6% | 0.3646 | 23 | Add new microsoft office proficiency content |
| **ADD** | restful api development | 28 | 1.5% | 0.3613 | 25 | Add new restful api development content |
| **ADD** | technical support | 27 | 1.4% | 0.3601 | 25 | Add new technical support content |
| **ADD** | git version control | 24 | 1.1% | 0.3573 | 26 | Add new git version control content |
| **ADD** | ci/cd pipelines | 26 | 1.1% | 0.3549 | 27 | Add new ci/cd pipelines content |
| **ADD** | postgresql database | 27 | 1.0% | 0.3523 | 28 | Add postgresql database content building on Data Mining |
| skip | code reviews | 18 | 0.9% | 0.3506 | 29 | Add new code reviews content |
| skip | system integration | 18 | 0.9% | 0.3505 | 29 | Add new system integration content |
| skip | incident response | 19 | 0.8% | 0.3486 | 30 | Add new incident response content |
| skip | api integration | 18 | 0.5% | 0.3478 | 30 | Add new api integration content |

Not selected, and why:

- **code reviews** — only closes 0.9% of the gap (coverage after the plan 0.67)
- **system integration** — absorbed by system maintenance (coverage after the plan 0.77)
- **incident response** — only closes 0.8% of the gap (coverage after the plan 0.66)
- **api integration** — absorbed by restful api development (coverage after the plan 0.78)

### Gaps with no verified host

No existing CLO shows verified evidence of teaching these — they need new content rather than an extension.

- **technical documentation** — 49 jobs, coverage 0.67 (nearest: It Project Management)
- **system maintenance** — 34 jobs, coverage 0.71 (nearest: Information Security)
- **restful api development** — 28 jobs, coverage 0.68 (nearest: Cloud Computing)
- **git version control** — 24 jobs, coverage 0.64 (nearest: It Project Management)
- **technical support** — 27 jobs, coverage 0.68 (nearest: Information Security)
- **ci/cd pipelines** — 26 jobs, coverage 0.69 (nearest: Introduction To Artificial Intelligence)
- **postgresql database** — 27 jobs, coverage 0.72 (nearest: Data Mining)
- **microsoft office proficiency** — 18 jobs, coverage 0.61 (nearest: Data Mining)
- **incident response** — 19 jobs, coverage 0.66 (nearest: Network Security)
- **api integration** — 18 jobs, coverage 0.65 (nearest: Cloud Computing)

---

Generated by `gen_report_data.py`. `report.json` in this directory holds the full machine-readable report.

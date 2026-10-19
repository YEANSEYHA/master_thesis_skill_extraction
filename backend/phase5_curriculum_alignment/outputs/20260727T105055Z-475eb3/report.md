# Phase 6 gap analysis — 20260727T105055Z-475eb3

**V8 tab · cambodia · 25 saved courses · Y3+Y4+Y5**

| | |
|---|---|
| ran | 2026-07-27T10:50:55+00:00 · 14.6 s |
| code | git 32fab9b4 |
| config hash | `475eb38e6ff0a1eaabd0f377e56dcf53ca500fe19beb84aca06a771af0a90eb6` |
| input hash | `a7c8b24affb0dc5a8732361597373687762e108a9966d11150a59bc76cab04d9` |
| cohort | Y3+Y4+Y5 |

## Inputs

**Market** — `cambodia` · `stage2_result.json` · 254 jobs in segment · segment mode `category`

**Curriculum** — 25 courses (db) · 590 CLO skills · cohort **Y3+Y4+Y5**

| course | file | CLO skills |
|---|---|---|
| Networks Ii | `10_networks_ii.json` | 8 |
| Systems And Networks Administration | `11_systems_and_networks_administration.json` | 31 |
| Advanced Database And Database Management Systems | `12_advanced_database_and_database_management_systems.json` | 14 |
| Distributed Systems | `13_distributed_systems.json` | 16 |
| Internet Programming Ii | `14_internet_programming_ii.json` | 26 |
| Introduction To Mobile Application Development | `15_introduction_to_mobile_application_development.json` | 11 |
| Network Design | `16_network_design.json` | 22 |
| Software Development And It Operations Devops | `17_software_development_and_it_operations_devops.json` | 18 |
| Cloud Computing | `18_cloud_computing.json` | 18 |
| Introduction To Artificial Intelligence | `19_introduction_to_artificial_intelligence.json` | 40 |
| Information System Analysis And Design | `1_information_system_analysis_and_design.json` | 20 |
| Natural Language Processing | `20_natural_language_processing.json` | 19 |
| Data Mining | `21_data_mining.json` | 55 |
| Image Processing | `22_image_processing.json` | 41 |
| Information Security | `23_information_security.json` | 14 |
| Network Security | `24_network_security.json` | 21 |
| It Project Management | `25_it_project_management.json` | 18 |
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

KL divergence **0.3689** · covered **25 / 30**

### Top skills by demand

| # | skill | jobs | coverage | status | best CLO match |
|---:|---|---:|---:|---|---|
| 1 | technical documentation | 49 | 0.78 | ✓ covered | Information System Analysis And Design — Data Integrity and Data Quality Documentation |
| 2 | database management | 42 | 0.91 | ✓ covered | Information System Analysis And Design — Database Management Systems (DBMS) |
| 3 | java programming | 41 | 0.87 | ✓ covered | Object Oriented Programming — Java |
| 4 | project management | 37 | 0.87 | ✓ covered | It Project Management — Project Team Management |
| 5 | network troubleshooting | 35 | 1.00 | ✓ covered | Introduction To Computer Systems And Networks — Network Troubleshooting |
| 6 | system maintenance | 34 | 0.82 | ✓ covered | Object Oriented Programming — Software Application Maintenance |
| 7 | mysql database | 30 | 0.78 | ✓ covered | Data Mining — Database Queries |
| 8 | restful api development | 28 | 0.76 | ✓ covered | Internet Programming Ii — Application Programming Interfaces (APIs) |
| 9 | postgresql database | 27 | 0.72 | ◑ borderline | Data Mining — Database Queries |
| 10 | technical support | 27 | 0.70 | ◑ borderline | Network Design — Network Maintenance and Troubleshooting |
| 11 | ci/cd pipelines | 26 | 0.96 | ✓ covered | Software Development And It Operations Devops — CI/CD Pipeline Implementation |
| 12 | docker containerization | 24 | 0.89 | ✓ covered | Cloud Computing — Containerization |
| 13 | git version control | 24 | 0.82 | ✓ covered | Software Development And It Operations Devops — Git |
| 14 | firewall configuration | 23 | 1.00 | ✓ covered | Introduction To Computer Systems And Networks — Firewall Configuration |
| 15 | security compliance | 23 | 1.00 | ✓ covered | Information Security — Security Compliance |
| 16 | server management | 21 | 0.82 | ✓ covered | Network Design — Network Management |
| 17 | aws cloud platforms | 19 | 0.82 | ✓ covered | Cloud Computing — AWS |
| 18 | incident response | 19 | 0.68 | ✗ gap | Software Development And It Operations Devops — Alerting |
| 19 | security monitoring | 19 | 0.86 | ✓ covered | Network Design — Network Monitoring |
| 20 | api integration | 18 | 0.83 | ✓ covered | Internet Programming Ii — Application Programming Interfaces (APIs) |
| 21 | code reviews | 18 | 0.68 | ✗ gap | Research Methodology — Literature Review |
| 22 | data analysis | 18 | 1.00 | ✓ covered | Research Methodology — Data Analysis |
| 23 | it infrastructure management | 18 | 0.81 | ✓ covered | Network Design — Network Management |
| 24 | microsoft office proficiency | 18 | 0.64 | ✗ gap | Information System Analysis And Design — Information System Components |
| 25 | software development | 18 | 0.74 | ✓ covered | Object Oriented Programming — Software Requirements Analysis |
| 26 | system integration | 18 | 0.82 | ✓ covered | Software Development And It Operations Devops — Continuous Integration Tools Setup and Configuration |
| 27 | vmware virtualization | 18 | 0.94 | ✓ covered | Systems And Networks Administration — Virtualization With VMware Workstation |
| 28 | performance tuning | 17 | 0.76 | ✓ covered | Distributed Systems — Distributed System Performance Analysis |
| 29 | disaster recovery planning | 16 | 1.00 | ✓ covered | Network Security — Disaster Recovery Planning |
| 30 | network administration | 16 | 0.89 | ✓ covered | Network Design — Network Management |

### Optimized skill plan

Greedy facility-location, stop at 80% of achievable gain → **add 4 of 5** gaps. KL 0.3689 → 0.3588

| | skill | jobs | closes | KL after | covered after | action |
|---|---|---:|---:|---:|---:|---|
| **ADD** | technical support | 27 | 1.9% | 0.3669 | 26 | Extend Systems And Networks Administration with explicit technical support content |
| **ADD** | microsoft office proficiency | 18 | 1.9% | 0.3649 | 27 | Add new microsoft office proficiency content |
| **ADD** | postgresql database | 27 | 1.2% | 0.3623 | 28 | Add postgresql database content building on Data Mining |
| **ADD** | code reviews | 18 | 1.0% | 0.3605 | 29 | Add new code reviews content |
| skip | incident response | 19 | 0.9% | 0.3588 | 30 | Add incident response content building on Software Development And It Operations Devops |

Not selected, and why:

- **incident response** — only closes 0.9% of the gap (coverage after the plan 0.68)

### Gaps with no verified host

No existing CLO shows verified evidence of teaching these — they need new content rather than an extension.

- **postgresql database** — 27 jobs, coverage 0.72 (nearest: Data Mining)
- **microsoft office proficiency** — 18 jobs, coverage 0.64 (nearest: Information System Analysis And Design)
- **incident response** — 19 jobs, coverage 0.68 (nearest: Software Development And It Operations Devops)
- **code reviews** — 18 jobs, coverage 0.68 (nearest: Research Methodology)

---

Generated by `gen_report_data.py`. `report.json` in this directory holds the full machine-readable report.

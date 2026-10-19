# Phase 6 gap analysis — 20260727T104742Z-475eb3

**V8 tab · cambodia · 8 saved courses · Y4**

| | |
|---|---|
| ran | 2026-07-27T10:47:42+00:00 · 13.5 s |
| code | git 32fab9b4 |
| config hash | `475eb38e6ff0a1eaabd0f377e56dcf53ca500fe19beb84aca06a771af0a90eb6` |
| input hash | `54f4eaed472c5400ba773b68cb84b5ecef873032777441966f1e0edeb7baf721` |
| cohort | Y4 |

## Inputs

**Market** — `cambodia` · `stage2_result.json` · 254 jobs in segment · segment mode `category`

**Curriculum** — 8 courses (db) · 146 CLO skills · cohort **Y4**

| course | file | CLO skills |
|---|---|---|
| Networks Ii | `1_networks_ii.json` | 8 |
| Systems And Networks Administration | `2_systems_and_networks_administration.json` | 31 |
| Advanced Database And Database Management Systems | `3_advanced_database_and_database_management_systems.json` | 14 |
| Distributed Systems | `4_distributed_systems.json` | 16 |
| Internet Programming Ii | `5_internet_programming_ii.json` | 26 |
| Introduction To Mobile Application Development | `6_introduction_to_mobile_application_development.json` | 11 |
| Network Design | `7_network_design.json` | 22 |
| Software Development And It Operations Devops | `8_software_development_and_it_operations_devops.json` | 18 |

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

KL divergence **0.3771** · covered **17 / 30**

### Top skills by demand

| # | skill | jobs | coverage | status | best CLO match |
|---:|---|---:|---:|---|---|
| 1 | technical documentation | 49 | 0.69 | ✗ gap | Software Development And It Operations Devops — Monitoring and Logging Strategies |
| 2 | database management | 42 | 0.86 | ✓ covered | Advanced Database And Database Management Systems — Database Transaction Management |
| 3 | java programming | 41 | 0.68 | ✗ gap | Internet Programming Ii — Application Programming Interfaces (APIs) |
| 4 | project management | 37 | 0.70 | ◑ borderline | Network Design — Network Management |
| 5 | network troubleshooting | 35 | 1.00 | ✓ covered | Systems And Networks Administration — Network Troubleshooting |
| 6 | system maintenance | 34 | 0.79 | ✓ covered | Network Design — Network Maintenance and Troubleshooting |
| 7 | mysql database | 30 | 0.71 | ◑ borderline | Internet Programming Ii — Database Seeding |
| 8 | restful api development | 28 | 0.76 | ✓ covered | Internet Programming Ii — Application Programming Interfaces (APIs) |
| 9 | postgresql database | 27 | 0.67 | ✗ gap | Internet Programming Ii — Database Migrations |
| 10 | technical support | 27 | 0.70 | ◑ borderline | Network Design — Network Maintenance and Troubleshooting |
| 11 | ci/cd pipelines | 26 | 0.96 | ✓ covered | Software Development And It Operations Devops — CI/CD Pipeline Implementation |
| 12 | docker containerization | 24 | 0.88 | ✓ covered | Software Development And It Operations Devops — Docker |
| 13 | git version control | 24 | 0.82 | ✓ covered | Software Development And It Operations Devops — Git |
| 14 | firewall configuration | 23 | 0.85 | ✓ covered | Systems And Networks Administration — Firewall Rules |
| 15 | security compliance | 23 | 0.73 | ✓ covered | Distributed Systems — Security Models |
| 16 | server management | 21 | 0.82 | ✓ covered | Network Design — Network Management |
| 17 | aws cloud platforms | 19 | 0.68 | ✗ gap | Systems And Networks Administration — Virtualization With VMware Workstation |
| 18 | incident response | 19 | 0.68 | ✗ gap | Software Development And It Operations Devops — Alerting |
| 19 | security monitoring | 19 | 0.86 | ✓ covered | Network Design — Network Monitoring |
| 20 | api integration | 18 | 0.83 | ✓ covered | Internet Programming Ii — Application Programming Interfaces (APIs) |
| 21 | code reviews | 18 | 0.67 | ✗ gap | Software Development And It Operations Devops — Infrastructure as Code (IaC) |
| 22 | data analysis | 18 | 0.70 | ◑ borderline | Advanced Database And Database Management Systems — Data Warehousing |
| 23 | it infrastructure management | 18 | 0.81 | ✓ covered | Network Design — Network Management |
| 24 | microsoft office proficiency | 18 | 0.60 | ✗ gap | Internet Programming Ii — Integration Testing |
| 25 | software development | 18 | 0.72 | ◑ borderline | Introduction To Mobile Application Development — Hybrid App Development |
| 26 | system integration | 18 | 0.82 | ✓ covered | Software Development And It Operations Devops — Continuous Integration Tools Setup and Configuration |
| 27 | vmware virtualization | 18 | 0.94 | ✓ covered | Systems And Networks Administration — Virtualization With VMware Workstation |
| 28 | performance tuning | 17 | 0.76 | ✓ covered | Distributed Systems — Distributed System Performance Analysis |
| 29 | disaster recovery planning | 16 | 0.70 | ◑ borderline | Systems And Networks Administration — Network Service Backup and Recovery Strategies |
| 30 | network administration | 16 | 0.89 | ✓ covered | Network Design — Network Management |

### Optimized skill plan

Greedy facility-location, stop at 80% of achievable gain → **add 9 of 13** gaps. KL 0.3771 → 0.3451

| | skill | jobs | closes | KL after | covered after | action |
|---|---|---:|---:|---:|---:|---|
| **ADD** | technical documentation | 49 | 3.1% | 0.3713 | 19 | Add new technical documentation content |
| **ADD** | java programming | 41 | 2.3% | 0.3662 | 20 | Add new java programming content |
| **ADD** | project management | 37 | 1.9% | 0.3629 | 21 | Add new project management content |
| **ADD** | microsoft office proficiency | 18 | 1.9% | 0.361 | 22 | Add new microsoft office proficiency content |
| **ADD** | mysql database | 30 | 1.7% | 0.3567 | 24 | Add mysql database content building on Advanced Database And Database Management Systems |
| **ADD** | software development | 18 | 1.6% | 0.3551 | 25 | Add new software development content |
| **ADD** | technical support | 27 | 1.4% | 0.3539 | 25 | Extend Systems And Networks Administration with explicit technical support content |
| **ADD** | data analysis | 18 | 1.0% | 0.3528 | 26 | Add new data analysis content |
| **ADD** | aws cloud platforms | 19 | 0.9% | 0.3508 | 27 | Add new aws cloud platforms content |
| skip | code reviews | 18 | 0.9% | 0.3494 | 28 | Add new code reviews content |
| skip | incident response | 19 | 0.8% | 0.3479 | 29 | Add incident response content building on Software Development And It Operations Devops |
| skip | postgresql database | 27 | 0.7% | 0.3462 | 29 | Add new postgresql database content |
| skip | disaster recovery planning | 16 | 0.7% | 0.3451 | 30 | Add new disaster recovery planning content |

Not selected, and why:

- **code reviews** — only closes 0.9% of the gap (coverage after the plan 0.67)
- **incident response** — only closes 0.8% of the gap (coverage after the plan 0.68)
- **postgresql database** — absorbed by mysql database (coverage after the plan 0.80)
- **disaster recovery planning** — only closes 0.7% of the gap (coverage after the plan 0.70)

### Gaps with no verified host

No existing CLO shows verified evidence of teaching these — they need new content rather than an extension.

- **technical documentation** — 49 jobs, coverage 0.69 (nearest: Software Development And It Operations Devops)
- **java programming** — 41 jobs, coverage 0.68 (nearest: Internet Programming Ii)
- **project management** — 37 jobs, coverage 0.70 (nearest: Network Design)
- **postgresql database** — 27 jobs, coverage 0.67 (nearest: Internet Programming Ii)
- **mysql database** — 30 jobs, coverage 0.71 (nearest: Internet Programming Ii)
- **microsoft office proficiency** — 18 jobs, coverage 0.60 (nearest: Internet Programming Ii)
- **aws cloud platforms** — 19 jobs, coverage 0.68 (nearest: Systems And Networks Administration)
- **incident response** — 19 jobs, coverage 0.68 (nearest: Software Development And It Operations Devops)
- **code reviews** — 18 jobs, coverage 0.67 (nearest: Software Development And It Operations Devops)
- **data analysis** — 18 jobs, coverage 0.70 (nearest: Advanced Database And Database Management Systems)

---

Generated by `gen_report_data.py`. `report.json` in this directory holds the full machine-readable report.

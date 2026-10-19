#!/usr/bin/env python3
"""
Data Analysis Tool for LLM Extraction Results
Usage: python visual_data.py <path_to_json_file>
Example: python visual_data.py outputs/whole_html_qwen3-coder_30b_20260203_162316.json
"""

import json
import sys
from collections import Counter
from datetime import datetime
from typing import List, Dict


def load_data(filepath: str) -> List[Dict]:
    """Load JSON data from file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file: {e}")
        sys.exit(1)


def print_header(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def analyze_completeness(data: List[Dict]) -> Dict:
    """Analyze data completeness and quality."""
    total = len(data)

    stats = {
        'total_records': total,
        'with_job_title': sum(1 for d in data if d.get('job_title')),
        'with_job_description': sum(1 for d in data if d.get('job_description')),
        'with_job_requirements': sum(1 for d in data if d.get('job_requirements')),
        'with_salary': sum(1 for d in data if d.get('salary')),
        'with_skills': sum(1 for d in data if d.get('skills') and len(d['skills']) > 0),
        'with_errors': sum(1 for d in data if d.get('error')),
        'complete_records': sum(1 for d in data if
            d.get('job_title') and
            d.get('job_description') and
            d.get('job_requirements') and
            d.get('skills') and len(d['skills']) > 0
        )
    }

    return stats


def analyze_skills(data: List[Dict]) -> Dict:
    """Analyze skills distribution."""
    all_skills = []
    skills_per_job = []

    for record in data:
        skills = record.get('skills', [])
        if skills:
            all_skills.extend(skills)
            skills_per_job.append(len(skills))

    skill_counts = Counter(all_skills)

    return {
        'total_skills': len(all_skills),
        'unique_skills': len(skill_counts),
        'top_skills': skill_counts.most_common(15),
        'avg_skills_per_job': sum(skills_per_job) / len(skills_per_job) if skills_per_job else 0,
        'max_skills': max(skills_per_job) if skills_per_job else 0,
        'min_skills': min([s for s in skills_per_job if s > 0]) if skills_per_job else 0
    }


def analyze_sources(data: List[Dict]) -> Dict:
    """Analyze source files."""
    sources = Counter(d.get('source_file', 'unknown') for d in data)
    domains = Counter()

    for source in sources.keys():
        if '_' in source:
            domain = source.split('_')[-1].replace('.html', '')
            domains[domain] += 1

    return {
        'total_sources': len(sources),
        'domains': domains.most_common(),
        'files_per_domain': dict(domains)
    }


def find_issues(data: List[Dict]) -> List[Dict]:
    """Find potential data quality issues."""
    issues = []

    for i, record in enumerate(data):
        source = record.get('source_file', f'record_{i}')

        # Check for errors
        if record.get('error'):
            issues.append({
                'source': source,
                'type': 'JSON Parse Error',
                'details': record.get('error')
            })

        # Check for generic/placeholder descriptions
        desc = record.get('job_description', '')
        if desc and ('Contact Employer' in desc or 'Interested in this position' in desc):
            issues.append({
                'source': source,
                'type': 'Generic Description',
                'details': f'Description: "{desc[:50]}..."'
            })

        # Check for missing critical fields
        if not record.get('job_title'):
            issues.append({
                'source': source,
                'type': 'Missing Job Title',
                'details': 'No job title extracted'
            })

        # Check for invented requirements
        req = record.get('job_requirements', '')
        if req and 'No specific requirements mentioned' in req:
            issues.append({
                'source': source,
                'type': 'Invented Requirement',
                'details': 'Model added placeholder text'
            })

    return issues


def print_analysis(filepath: str):
    """Main analysis function."""
    print(f"\n{'#'*70}")
    print(f"# LLM DATA EXTRACTION ANALYSIS")
    print(f"# File: {filepath}")
    print(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")

    # Load data
    data = load_data(filepath)

    # 1. Overall Statistics
    print_header("📊 OVERALL STATISTICS")
    stats = analyze_completeness(data)
    print(f"  Total Records: {stats['total_records']}")
    print(f"\n  Field Completeness:")
    print(f"    • Job Title:       {stats['with_job_title']:>4} / {stats['total_records']} ({stats['with_job_title']/stats['total_records']*100:.1f}%)")
    print(f"    • Description:     {stats['with_job_description']:>4} / {stats['total_records']} ({stats['with_job_description']/stats['total_records']*100:.1f}%)")
    print(f"    • Requirements:    {stats['with_job_requirements']:>4} / {stats['total_records']} ({stats['with_job_requirements']/stats['total_records']*100:.1f}%)")
    print(f"    • Salary:          {stats['with_salary']:>4} / {stats['total_records']} ({stats['with_salary']/stats['total_records']*100:.1f}%)")
    print(f"    • Skills:          {stats['with_skills']:>4} / {stats['total_records']} ({stats['with_skills']/stats['total_records']*100:.1f}%)")
    print(f"\n  Quality Metrics:")
    print(f"    • Complete Records: {stats['complete_records']} ({stats['complete_records']/stats['total_records']*100:.1f}%)")
    print(f"    • Records w/ Errors: {stats['with_errors']}")

    # 2. Skills Analysis
    print_header("🎯 SKILLS ANALYSIS")
    skills_data = analyze_skills(data)
    print(f"  Total Skills Extracted: {skills_data['total_skills']}")
    print(f"  Unique Skills: {skills_data['unique_skills']}")
    print(f"  Average Skills per Job: {skills_data['avg_skills_per_job']:.1f}")
    print(f"  Skills Range: {skills_data['min_skills']} - {skills_data['max_skills']} per job")

    print(f"\n  📈 Top 15 Most Common Skills:")
    for i, (skill, count) in enumerate(skills_data['top_skills'], 1):
        bar_length = int((count / skills_data['top_skills'][0][1]) * 30)
        bar = '█' * bar_length
        print(f"    {i:2d}. {skill:<30} {bar} {count:>3}")

    # 3. Source Analysis
    print_header("📁 SOURCE FILES ANALYSIS")
    source_data = analyze_sources(data)
    print(f"  Total Source Files: {source_data['total_sources']}")
    print(f"  Domains Found: {len(source_data['domains'])}")

    print(f"\n  📊 Records by Domain:")
    for domain, count in source_data['domains']:
        percentage = (count / stats['total_records']) * 100
        bar_length = int(percentage / 100 * 30)
        bar = '█' * bar_length
        print(f"    {domain:<20} {bar} {count:>3} ({percentage:.1f}%)")

    # 4. Data Quality Issues
    print_header("⚠️  DATA QUALITY ISSUES")
    issues = find_issues(data)

    if not issues:
        print("  ✓ No major issues detected!")
    else:
        issue_types = Counter(issue['type'] for issue in issues)
        print(f"  Total Issues Found: {len(issues)}")
        print(f"\n  Issues by Type:")
        for issue_type, count in issue_types.most_common():
            print(f"    • {issue_type}: {count}")

        print(f"\n  Sample Issues (first 10):")
        for i, issue in enumerate(issues[:10], 1):
            print(f"    {i}. [{issue['type']}] {issue['source']}")
            print(f"       → {issue['details']}")

    # 5. Sample Records
    print_header("📝 SAMPLE RECORDS")
    print(f"  Showing 3 sample records:\n")

    # Show one complete, one partial, and one with issues if available
    complete = [d for d in data if d.get('job_title') and d.get('job_description') and d.get('skills') and len(d['skills']) > 0]
    partial = [d for d in data if not (d.get('job_title') and d.get('job_description') and d.get('skills'))]

    samples = []
    if complete:
        samples.append(('Complete Record', complete[0]))
    if len(complete) > 1:
        samples.append(('Complete Record', complete[1]))
    if partial:
        samples.append(('Incomplete Record', partial[0]))

    for label, record in samples[:3]:
        print(f"  ▸ {label}: {record.get('source_file', 'unknown')}")
        print(f"    Title: {record.get('job_title', 'N/A')}")
        desc = record.get('job_description') or 'N/A'
        desc_str = str(desc) if desc else 'N/A'
        print(f"    Description: {desc_str[:80]}{'...' if len(desc_str) > 80 else ''}")
        skills = record.get('skills', [])
        skills_str = ', '.join(skills[:5]) if skills else 'N/A'
        print(f"    Skills: {skills_str}{'...' if len(skills) > 5 else ''}")
        print()

    # 6. Summary & Recommendations
    print_header("💡 SUMMARY & RECOMMENDATIONS")

    success_rate = (stats['complete_records'] / stats['total_records']) * 100

    if success_rate >= 80:
        print("  ✅ Excellent extraction quality!")
    elif success_rate >= 60:
        print("  ⚠️  Good extraction quality, but room for improvement")
    else:
        print("  ❌ Low extraction quality - consider improving prompts or HTML cleaning")

    print(f"\n  Key Metrics:")
    print(f"    • Success Rate: {success_rate:.1f}%")
    print(f"    • Data Richness: {skills_data['avg_skills_per_job']:.1f} skills per job")
    print(f"    • Error Rate: {(stats['with_errors']/stats['total_records']*100):.1f}%")

    if len(issue_types) > 0:
        print(f"\n  ⚡ Actions Recommended:")
        if 'Generic Description' in issue_types:
            print(f"    • {issue_types['Generic Description']} records have generic descriptions - improve HTML cleaning")
        if 'Invented Requirement' in issue_types:
            print(f"    • {issue_types['Invented Requirement']} records have invented text - adjust prompt to avoid hallucination")
        if 'Missing Job Title' in issue_types:
            print(f"    • {issue_types['Missing Job Title']} records missing titles - check HTML structure")

    print(f"\n{'='*70}\n")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("\nUsage: python visual_data.py <path_to_json_file>")
        print("\nExample:")
        print("  python visual_data.py outputs/whole_html_qwen3-coder_30b_20260203_162316.json")
        print()
        sys.exit(1)

    filepath = sys.argv[1]
    print_analysis(filepath)


if __name__ == "__main__":
    main()

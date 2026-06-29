# label_tool.py
"""
Interactive CLI tool to grade a sample of candidates (0-4) for the 
Senior AI Engineer role. Saves progress to labeled_candidates.json.
"""

import json
from pathlib import Path
from features import is_honeypot


def print_candidate(c: dict, index: int, total: int):
    p = c.get('profile', {})
    skills = [s['name'] for s in c.get('skills', [])]
    career = c.get('career_history', [])
    signals = c.get('redrob_signals', {})
    
    print("\n" + "=" * 80)
    print(f" Candidate Profile [{index}/{total}]: {c.get('candidate_id', 'UNKNOWN')}")
    print("=" * 80)
    print(f"Headline:       {p.get('headline', 'N/A')}")
    print(f"Current Role:   {p.get('current_title', 'N/A')} at {p.get('current_company', 'N/A')}")
    print(f"Experience:     {p.get('years_of_experience', 'N/A')} years")
    print(f"Location:       {p.get('location', 'N/A')}, {p.get('country', 'N/A')} (Relocate: {signals.get('willing_to_relocate', 'N/A')})")
    print(f"Expected Sal:   {signals.get('expected_salary_range_inr_lpa', 'N/A')} LPA")
    print(f"Notice Period:  {signals.get('notice_period_days', 'N/A')} days")
    
    print("\n--- Key Skills ---")
    print(", ".join(skills[:15]))
    
    print("\n--- Career History (Top 3) ---")
    for i, role in enumerate(career[:3]):
        desc = role.get('description', '')
        desc_snippet = (desc[:140] + "...") if len(desc) > 140 else desc
        print(f"• {role.get('title', 'N/A')} at {role.get('company', 'N/A')} ({role.get('duration_months', 0)} mo)")
        if desc_snippet:
            print(f"  \"{desc_snippet}\"")
            
    print("\n--- Summary ---")
    summary = p.get('summary', '')
    print(summary if summary else "No summary available.")
    print("-" * 80)


def main():
    # Find candidates source
    candidates_path = Path("sample_candidates.json")
    if not candidates_path.exists():
        candidates_path = Path("candidates.jsonl")
        
    if not candidates_path.exists():
        print("Error: Could not find sample_candidates.json or candidates.jsonl")
        return
        
    print(f"Loading candidates from {candidates_path}...")
    if str(candidates_path).endswith('.json'):
        with open(candidates_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            candidates = data if isinstance(data, list) else [data]
    else:
        candidates = []
        with open(candidates_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    candidates.append(json.loads(line))
                    
    # Filter out honeypots to avoid wasting time labeling them
    valid_candidates = [c for c in candidates if not is_honeypot(c)]
    print(f"Loaded {len(candidates)} total. Found {len(valid_candidates)} non-honeypot candidates available for grading.")
    
    # Load existing labels
    labels_file = Path("labeled_candidates.json")
    labels = {}
    if labels_file.exists():
        try:
            with open(labels_file, "r", encoding="utf-8") as f:
                labels = json.load(f)
            print(f"Loaded {len(labels)} existing labels from labeled_candidates.json")
        except json.JSONDecodeError:
            print("Warning: labeled_candidates.json was empty or corrupted. Starting fresh.")
            
    total = len(valid_candidates)
    graded_count = sum(1 for c in valid_candidates if c.get('candidate_id') in labels)
    
    for idx, c in enumerate(valid_candidates, 1):
        cid = c.get('candidate_id')
        if not cid:
            continue
            
        if cid in labels:
            continue
            
        print_candidate(c, idx, total)
        print("Relevance Grading Rubric:")
        print("  [4] Outstanding Hire  - Perfect technical fit, senior target YoE, India-based (Top-10 candidates)")
        print("  [3] Strong Hire       - Very strong skills & experience match, good background (Top-50 candidates)")
        print("  [2] Acceptable        - Meets core skills, worth interviewing")
        print("  [1] Weak Fit          - Lacks deep core skills or experience, possible but unlikely")
        print("  [0] Reject            - Honeypot, wrong domain (CV/Speech/Robotics), or consulting-only background")
        print("  [s] Skip candidate")
        print("  [q] Quit and save")
        
        while True:
            choice = input(f"\nEnter grade for {cid} (0-4, s, q): ").strip().lower()
            if choice == 'q':
                print(f"\nExiting. Total labeled candidates: {len(labels)}")
                return
            if choice == 's':
                print("Skipped candidate.")
                break
            if choice in ('0', '1', '2', '3', '4'):
                grade = int(choice)
                labels[cid] = grade
                # Save incrementally
                with open(labels_file, "w", encoding="utf-8") as f:
                    json.dump(labels, f, indent=2)
                print(f"Saved grade {grade} for {cid}.")
                break
            print("Invalid input. Please enter 0, 1, 2, 3, 4, s, or q.")
            
    print(f"\nAll candidates graded! Total labeled: {len(labels)}")


if __name__ == "__main__":
    main()

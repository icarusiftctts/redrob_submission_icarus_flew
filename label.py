# label.py
"""
Interactive labeling tool.
Labels: 3=definitely hire  2=probably hire  1=borderline  0=reject

Usage:
    python label.py                              # labels sample_candidates.json
    python label.py --extra candidates.jsonl     # also labels from full pool
"""

import argparse
import gzip
import json
import random
from pathlib import Path


def load_sample(path='sample_candidates.json') -> list:
    with open(path) as f:
        return json.load(f)


def load_extra_from_jsonl(path: str, skip_ids: set, limit: int = 300) -> list:
    extra = []
    path_str = str(path)
    if path_str.endswith('.gz'):
        opener = lambda: gzip.open(path_str, 'rt', encoding='utf-8')
    else:
        opener = lambda: open(path_str, 'r', encoding='utf-8')
    
    count = 0
    with opener() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if c['candidate_id'] in skip_ids:
                continue
            count += 1
            if len(extra) < limit:
                extra.append(c)
            else:
                s = random.randint(0, count - 1)
                if s < limit:
                    extra[s] = c
    return extra


def show_candidate(c: dict) -> None:
    p   = c['profile']
    sig = c['redrob_signals']
    ca  = c.get('career_history', [])
    sk  = c.get('skills', [])

    print(f"\n{'='*60}")
    print(f"ID: {c['candidate_id']}")
    print(f"Title:  {p['current_title']}  |  {p['years_of_experience']:.0f} yr  |  {p['location']}")
    print(f"OTW:    {sig['open_to_work_flag']}  |  "
          f"Notice: {sig['notice_period_days']}d  |  "
          f"Response: {sig['recruiter_response_rate']:.0%}  |  "
          f"Active: {sig['last_active_date']}")
    print(f"Skills: {[s['name'] for s in sk[:24]]}")
    print("Career:")
    for r in ca[:12]:
        print(f"  {r['title']:35s}  @  {r['company']:20s}  ({r['duration_months']}mo)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extra', default=None,
                        help='Path to candidates.jsonl/candidates.jsonl.gz for extra candidates')
    parser.add_argument('--limit', type=int, default=300,
                        help='Max candidates to load from --extra')
    parser.add_argument('--batch-size', '-b', type=int, default=None,
                        help='Number of candidates to label in this session before auto-exiting')
    parser.add_argument('--no-shuffle', action='store_true',
                        help='Disable shuffling of candidates before labeling')
    args = parser.parse_args()

    label_file = Path('labeled_candidates.json')
    labels: dict = {}
    if label_file.exists():
        labels = json.load(open(label_file))
        print(f'Resuming: {len(labels)} already labeled.')

    # Build candidate list
    candidates = load_sample()
    if args.extra:
        extra_path = Path(args.extra)
        if not extra_path.exists():
            if args.extra.endswith('.gz') and Path(args.extra[:-3]).exists():
                extra_path = Path(args.extra[:-3])
            elif not args.extra.endswith('.gz') and Path(args.extra + '.gz').exists():
                extra_path = Path(args.extra + '.gz')
            else:
                print(f"WARNING: Extra candidates file {args.extra} not found. Skipping.")
                extra_path = None
        
        if extra_path:
            extra = load_extra_from_jsonl(extra_path, set(labels), args.limit)
            candidates = candidates + extra
            print(f'Total candidates loaded: {len(candidates)}')

    # Filter out already labeled candidates
    candidates = [c for c in candidates if c['candidate_id'] not in labels]

    # Shuffle candidates by default
    if not args.no_shuffle:
        random.shuffle(candidates)
        print("Shuffled candidate pool to prevent bias.")

    print(f'Total candidates to label in this session: {len(candidates)}')

    labeled_this_session = 0
    print(f"Starting labeling session. Type 'q' or use Ctrl+C to quit/pause at any time.")
    if args.batch_size:
        print(f"Session limit set to {args.batch_size} candidates.")

    try:
        for c in candidates:
            if args.batch_size is not None and labeled_this_session >= args.batch_size:
                print(f"\nReached batch size limit of {args.batch_size} candidates. Auto-saving and exiting.")
                break

            cid = c['candidate_id']
            if cid in labels:
                continue

            show_candidate(c)
            label = input('\n  Label  3=hire  2=maybe  1=weak  0=reject  s=skip  q=quit: ').strip()

            if label == 'q':
                break
            if label == 's':
                continue
            if label in ('0', '1', '2', '3'):
                labels[cid] = int(label)
                json.dump(labels, open(label_file, 'w'))
                labeled_this_session += 1
    except KeyboardInterrupt:
        print("\n\nLabeling session interrupted by user. Progress saved.")

    print(f'\nSession: labeled {labeled_this_session} new candidates.')
    print(f'Total labeled: {len(labels)}  →  labeled_candidates.json')

    # Show distribution
    if labels:
        from collections import Counter
        dist = Counter(labels.values())
        names = {3: 'hire', 2: 'maybe', 1: 'weak', 0: 'reject'}
        for lbl in sorted(dist):
            print(f"  Label {lbl} ({names[lbl]:6s}): {dist[lbl]}")


if __name__ == '__main__':
    main()

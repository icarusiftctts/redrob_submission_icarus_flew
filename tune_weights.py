# tune_weights.py
"""
Learn optimal feature weights from manually labeled candidates.

Usage:
    python tune_weights.py
    python tune_weights.py --labels labeled_candidates.json
"""

import argparse
import gzip
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from features import score_candidate, extract_features


def load_all_candidates(sample_path='sample_candidates.json',
                        jsonl_path='candidates.jsonl',
                        label_ids: set = None) -> dict:
    """Return {candidate_id: candidate} from sample + candidates in candidates.jsonl matching label_ids."""
    cands = {}
    if label_ids is None:
        label_ids = set()

    with open(sample_path) as f:
        for c in json.load(f):
            cands[c['candidate_id']] = c
            
    path_str = str(jsonl_path)
    if not Path(path_str).exists():
        if path_str.endswith('.gz') and Path(path_str[:-3]).exists():
            path_str = path_str[:-3]
        elif not path_str.endswith('.gz') and Path(path_str + '.gz').exists():
            path_str = path_str + '.gz'
            
    if Path(path_str).exists():
        if path_str.endswith('.gz'):
            opener = lambda: gzip.open(path_str, 'rt', encoding='utf-8')
        else:
            opener = lambda: open(path_str, 'r', encoding='utf-8')
        with opener() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                c = json.loads(line)
                cid = c['candidate_id']
                if cid in label_ids or cid in cands:
                    cands[cid] = c
    return cands


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels',      default='labeled_candidates.json')
    parser.add_argument('--precomputed', default='precomputed')
    parser.add_argument('--sample',      default='sample_candidates.json')
    parser.add_argument('--candidates',  default='candidates.jsonl')
    args = parser.parse_args()

    # ── Load labels ───────────────────────────────────────────────────────────
    if not Path(args.labels).exists():
        print(f"ERROR: Labels file {args.labels} not found. Please run label.py first.")
        return

    labels: dict = json.load(open(args.labels))
    print(f'Labels loaded: {len(labels)} candidates')

    # ── Load precomputed scores ───────────────────────────────────────────────
    pre = Path(args.precomputed)
    if not pre.exists():
        print(f"ERROR: Precomputed folder {pre} not found. Please run precompute.py first.")
        return

    id_to_idx    = pickle.load(open(pre / 'id_to_idx.pkl', 'rb'))
    sbert_scores = np.load(pre / 'sbert_scores.npy')
    bm25_scores  = np.load(pre / 'bm25_scores.npy')

    # ── Load candidate pool ───────────────────────────────────────────────────
    cand_map = load_all_candidates(args.sample, args.candidates, set(labels.keys()))
    print(f'Candidate pool loaded: {len(cand_map)} candidates')

    # ── Build feature matrix ──────────────────────────────────────────────────
    X_rows, y_rows, feature_names = [], [], None

    for cid, label in labels.items():
        if cid not in cand_map:
            continue
        c   = cand_map[cid]
        idx = id_to_idx.get(cid, 0)
        sb  = float(sbert_scores[min(idx, len(sbert_scores) - 1)]) if len(sbert_scores) > 0 else 0.0
        bm  = float(bm25_scores[min(idx, len(bm25_scores) - 1)]) if len(bm25_scores) > 0 else 0.0

        _, feats = score_candidate(c, sb, bm)
        if not feats:
            continue   # honeypot — skip

        if feature_names is None:
            feature_names = list(feats.keys())

        X_rows.append([feats[k] for k in feature_names])
        y_rows.append(label)

    if not X_rows:
        print("ERROR: No features extracted from labeled candidates. Ensure candidate IDs match.")
        return

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int32)
    print(f'\nFeature matrix: {X.shape}')
    print(f'Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}')

    # Check if we have more than one class to train classifier
    if len(np.unique(y)) < 2:
        print("ERROR: Need at least 2 different classes to optimize weights. Label more candidates with different ratings.")
        return

    # ── Train + cross-validate ────────────────────────────────────────────────
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    model = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    
    cv_folds = min(5, len(y))
    if cv_folds >= 2:
        cv    = cross_val_score(model, X_sc, y, cv=cv_folds, scoring='f1_weighted')
        print(f'\nCV F1 ({cv_folds}-fold): {cv.mean():.3f} ± {cv.std():.3f}   '
              f'(target > 0.60)')

    model.fit(X_sc, y)

    # ── Feature importance for the "hire" class (label = 3 or highest label available) ──────────────────
    highest_class_idx = -1
    coef   = model.coef_[highest_class_idx]   # coefficients for the highest class
    ranked = sorted(zip(feature_names, coef), key=lambda x: -abs(x[1]))

    print('\n── Feature importance for hire class ─────────────────────────')
    print(f'  {"Feature":<20}  Coef')
    for name, w in ranked:
        bar = '█' * int(abs(w) * 10)
        sign = '+' if w > 0 else '-'
        print(f'  {name:<20}  {sign}{abs(w):.3f}  {bar}')

    print('\n── Action ────────────────────────────────────────────────────')
    print('  In features.py, update WEIGHTS:')
    print('  • Increase weights for features with large POSITIVE coef')
    print('  • Decrease weights for features with large NEGATIVE coef')
    print('  • Remove features with |coef| < 0.05 (they add no signal)')

    # ── Save model ────────────────────────────────────────────────────────────
    out = Path(args.precomputed) / 'learned_model.pkl'
    pickle.dump({
        'model':    model,
        'scaler':   scaler,
        'features': feature_names,
    }, open(out, 'wb'))
    print(f'\nModel saved to {out}')
    print('To use in rank.py: load it and replace score_candidate() with')
    print('  model.predict_proba(scaler.transform([feature_vector]))[0][-1]')


if __name__ == '__main__':
    main()

# tune_weights.py
"""
Optimizes the feature weights in features.py based on user-provided relevance labels
stored in labeled_candidates.json. Uses SLSQP constrained optimization with a 
smooth pairwise ranking loss (RankNet style) regularized toward initial heuristic weights.
"""

import re
import json
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from features import (
    is_honeypot,
    extract_features,
    location_multiplier,
    yoe_multiplier,
    salary_multiplier,
    compute_behavioral_multiplier,
    WEIGHTS as INITIAL_WEIGHTS
)


def load_labeled_candidates(labels_path: Path, candidates_path: Path):
    """Loads labeled candidates and extracts their features/multipliers."""
    with open(labels_path, "r", encoding="utf-8") as f:
        labels_dict = json.load(f)
        
    if not labels_dict:
        print("No labels found in labeled_candidates.json. Please run label_tool.py first.")
        return []
        
    print(f"Loading candidates database from {candidates_path}...")
    if str(candidates_path).endswith('.json'):
        with open(candidates_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            candidates_all = data if isinstance(data, list) else [data]
    else:
        candidates_all = []
        with open(candidates_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    candidates_all.append(json.loads(line))
                    
    labeled_data = []
    
    # Precomputed files (if they exist)
    pre_dir = Path("precomputed")
    sbert_scores = np.zeros(len(candidates_all))
    bm25_scores = np.zeros(len(candidates_all))
    id_to_idx = {}
    
    if pre_dir.exists():
        import pickle
        try:
            id_to_idx = pickle.load(open(pre_dir / 'id_to_idx.pkl', 'rb'))
            sbert_scores = np.load(pre_dir / 'sbert_scores.npy')
            bm25_scores = np.load(pre_dir / 'bm25_scores.npy')
        except Exception as e:
            print(f"Warning: Failed to load precomputed scores: {e}. Using defaults.")

    for i, c in enumerate(candidates_all):
        cid = c.get('candidate_id')
        if cid in labels_dict:
            # Get precomputed scores if available
            idx = id_to_idx.get(cid, i)
            sb = float(sbert_scores[min(idx, len(sbert_scores)-1)]) if len(sbert_scores) > 0 else 0.0
            bm = float(bm25_scores[min(idx, len(bm25_scores)-1)]) if len(bm25_scores) > 0 else 0.0
            
            # Extract raw features and multipliers
            feats = extract_features(c, sb, bm)
            
            # Compute multipliers
            loc_m = location_multiplier(c.get('profile', {}), c.get('redrob_signals', {}))
            yoe_m = yoe_multiplier(c.get('profile', {}).get('years_of_experience', 0))
            sal_m = salary_multiplier(c.get('redrob_signals', {}))
            mult = compute_behavioral_multiplier(c.get('redrob_signals', {}))
            
            # Combine multipliers: Final score = (Base_Score + 0.15 * (mult - 1.0)) * loc_m * yoe_m * sal_m
            # Rewritten as: Base_Score * Multiplier_Product + Offset_Product
            multiplier_prod = loc_m * yoe_m * sal_m
            offset_prod = 0.15 * (mult - 1.0) * multiplier_prod
            
            labeled_data.append({
                'candidate_id': cid,
                'label': labels_dict[cid],
                'features': feats,
                'multiplier_prod': multiplier_prod,
                'offset_prod': offset_prod
            })
            
    print(f"Successfully compiled {len(labeled_data)} labeled candidates.")
    return labeled_data


def compute_scores(weights: dict, labeled_data: list):
    """Computes final score for each labeled candidate using the weights dict."""
    scores = []
    for item in labeled_data:
        base_score = sum(weights[k] * item['features'][k] for k in weights)
        final_score = base_score * item['multiplier_prod'] + item['offset_prod']
        scores.append(max(0.0, final_score))
    return np.array(scores)


def calculate_ndcg(labels: np.ndarray, scores: np.ndarray, k: int = 10):
    """Computes NDCG@K."""
    if len(labels) == 0:
        return 0.0
    
    # Sort scores descending
    order = np.argsort(scores)[::-1]
    sorted_labels = labels[order]
    
    # DCG
    dcg = sum((2 ** val - 1) / np.log2(r + 2) for r, val in enumerate(sorted_labels[:k]))
    
    # IDCG (ideal sorting)
    ideal_labels = np.sort(labels)[::-1]
    idcg = sum((2 ** val - 1) / np.log2(r + 2) for r, val in enumerate(ideal_labels[:k]))
    
    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_ranking(weights_dict: dict, labeled_data: list):
    """Evaluates the ranking performance (NDCG and Pairwise accuracy) of a set of weights."""
    labels = np.array([x['label'] for x in labeled_data])
    scores = compute_scores(weights_dict, labeled_data)
    
    ndcg_10 = calculate_ndcg(labels, scores, k=10)
    ndcg_all = calculate_ndcg(labels, scores, k=len(labels))
    
    # Pairwise accuracy
    correct_pairs = 0
    total_pairs = 0
    for i in range(len(labeled_data)):
        for j in range(len(labeled_data)):
            if labels[i] > labels[j]:
                total_pairs += 1
                if scores[i] > scores[j]:
                    correct_pairs += 1
                elif scores[i] == scores[j]:
                    correct_pairs += 0.5  # tie breaker penalty
                    
    pairwise_acc = (correct_pairs / total_pairs) if total_pairs > 0 else 1.0
    return ndcg_10, ndcg_all, pairwise_acc


def optimize_weights(labeled_data: list, initial_weights: dict, alpha: float = 0.5):
    """
    Optimizes weights by minimizing pairwise RankNet loss.
    Regularizes weights towards the initial heuristics.
    alpha: Regularization strength (higher = keep weights closer to initial heuristics).
    """
    keys = list(initial_weights.keys())
    w0_vec = np.array([initial_weights[k] for k in keys])
    
    # Construct feature matrix X (N x D)
    N = len(labeled_data)
    D = len(keys)
    X = np.zeros((N, D))
    mults = np.zeros(N)
    offsets = np.zeros(N)
    labels = np.zeros(N)
    
    for i, item in enumerate(labeled_data):
        labels[i] = item['label']
        mults[i] = item['multiplier_prod']
        offsets[i] = item['offset_prod']
        for j, k in enumerate(keys):
            X[i, j] = item['features'][k]
            
    # Compile pairs (i, j) where label[i] > label[j]
    pairs = []
    for i in range(N):
        for j in range(N):
            if labels[i] > labels[j]:
                pairs.append((i, j))
                
    if not pairs:
        print("Error: No distinct pairs found to rank (all labels are equal).")
        return initial_weights
        
    def objective(w):
        # Compute scores for all candidates
        # base_scores = X @ w
        # scores = base_scores * mults + offsets
        base_scores = np.dot(X, w)
        scores = base_scores * mults + offsets
        
        # Pairwise logistic loss (RankNet loss)
        loss = 0.0
        for i, j in pairs:
            diff = scores[i] - scores[j]
            # Use log(1 + exp(-diff)) safely to avoid overflow
            if diff > 20:
                loss += 0.0
            elif diff < -20:
                loss += -diff
            else:
                loss += np.log(1.0 + np.exp(-diff))
                
        # L2 Regularization towards initial weights
        reg = alpha * np.sum((w - w0_vec) ** 2)
        return loss + reg

    # Constraints: sum(w) = 1.0
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    # Bounds: 0 <= w_i <= 1.0
    bounds = [(0.0, 1.0) for _ in range(D)]
    
    print("\nRunning optimization...")
    res = minimize(objective, w0_vec, method='SLSQP', bounds=bounds, constraints=cons)
    
    if not res.success:
        print(f"Warning: Optimization did not converge cleanly: {res.message}")
        
    opt_w_vec = res.x
    # Normalize to ensure sum is exactly 1.0
    opt_w_vec = opt_w_vec / np.sum(opt_w_vec)
    
    opt_weights = {keys[i]: float(opt_w_vec[i]) for i in range(D)}
    return opt_weights


def update_features_file(new_weights: dict):
    """Replaces the WEIGHTS block in features.py with the optimized weights."""
    file_path = Path("features.py")
    if not file_path.exists():
        print("Error: features.py not found.")
        return False
        
    content = file_path.read_text(encoding="utf-8")
    
    # Construct the new WEIGHTS block string
    new_block = "WEIGHTS = {\n"
    # Group items just like in the original structure for clean formatting
    groups = {
        "Skills group": ['skill_core', 'retrieval', 'python_sk', 'skill_endorse', 'skill_dur', 'assess', 'github'],
        "Career group": ['title_curr', 'title_avg', 'consult', 'hop', 'prog', 'product', 'tenure'],
        "Semantic group": ['sbert', 'bm25', 'headline'],
        "Exp/Loc/Edu": ['yoe', 'loc', 'edu'],
        "New signals": ['cert', 'recruit']
    }
    
    for group_name, keys in groups.items():
        new_block += f"    # {group_name}\n"
        for k in keys:
            val = new_weights.get(k, 0.0)
            new_block += f"    '{k}':{val:10.4f},\n"
            
    new_block += "}"
    
    # Regex to find and replace WEIGHTS = { ... }
    pattern = re.compile(r"WEIGHTS = \{[^\}]+\}", re.MULTILINE)
    
    if not pattern.search(content):
        print("Error: Could not locate 'WEIGHTS = { ... }' in features.py.")
        return False
        
    updated_content = pattern.sub(new_block, content)
    file_path.write_text(updated_content, encoding="utf-8")
    print("\nSuccessfully updated features.py with optimized weights!")
    return True


def main():
    labels_path = Path("labeled_candidates.json")
    if not labels_path.exists():
        print("Error: labeled_candidates.json not found. Please label some candidates first using label_tool.py.")
        return
        
    candidates_path = Path("sample_candidates.json")
    if not candidates_path.exists():
        candidates_path = Path("candidates.jsonl")
        
    if not candidates_path.exists():
        print("Error: Could not find candidates source file.")
        return
        
    labeled_data = load_labeled_candidates(labels_path, candidates_path)
    if len(labeled_data) < 5:
        print("Warning: You have labeled fewer than 5 candidates. Weight tuning might overfit. Recommend labeling at least 10–20.")
        if len(labeled_data) == 0:
            return
            
    # Evaluate initial heuristic weights
    i_ndcg_10, i_ndcg_all, i_pair_acc = evaluate_ranking(INITIAL_WEIGHTS, labeled_data)
    
    print("\nInitial Heuristic Performance:")
    print(f"  NDCG@10:          {i_ndcg_10:.4f}")
    print(f"  NDCG@All:         {i_ndcg_all:.4f}")
    print(f"  Pairwise Accuracy: {i_pair_acc * 100:.2f}%")
    
    # Run optimization
    opt_weights = optimize_weights(labeled_data, INITIAL_WEIGHTS, alpha=1.0)
    
    # Evaluate optimized weights
    o_ndcg_10, o_ndcg_all, o_pair_acc = evaluate_ranking(opt_weights, labeled_data)
    
    print("\nOptimized Performance:")
    print(f"  NDCG@10:          {o_ndcg_10:.4f}")
    print(f"  NDCG@All:         {o_ndcg_all:.4f}")
    print(f"  Pairwise Accuracy: {o_pair_acc * 100:.2f}%")
    
    print("\nWeight Changes:")
    print(f"{'Feature':<15} | {'Initial':<10} | {'Optimized':<10} | {'Change':<10}")
    print("-" * 55)
    for k in INITIAL_WEIGHTS:
        w_init = INITIAL_WEIGHTS[k]
        w_opt = opt_weights[k]
        diff = w_opt - w_init
        print(f"{k:<15} | {w_init:<10.4f} | {w_opt:<10.4f} | {diff:+.4f}")
        
    # Prompt to update features.py
    ans = input("\nWould you like to write these optimized weights to features.py? (y/n): ").strip().lower()
    if ans == 'y':
        update_features_file(opt_weights)
    else:
        print("Optimized weights were not written to file.")


if __name__ == "__main__":
    main()

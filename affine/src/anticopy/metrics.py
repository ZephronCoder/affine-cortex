"""
Logprob similarity metrics for copy detection.

Three complementary signals:
  1. cosine_similarity  – logprob vector correlation (primary signal)
  2. js_divergence      – top-k probability distribution distance
  3. token_agreement    – fraction of positions with same greedy token
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


def batch_cosine_similarity(
    vecs_a: np.ndarray,  # (n_tasks, n_tokens)
    vecs_b: np.ndarray,  # (n_tasks, n_tokens)
) -> np.ndarray:
    """Compute per-task cosine similarity between two logprob matrices.

    Returns shape (n_tasks,) array of cosine similarities.
    Tasks where either norm is zero are assigned NaN.
    """
    norms_a = np.linalg.norm(vecs_a, axis=1, keepdims=True)  # (n_tasks, 1)
    norms_b = np.linalg.norm(vecs_b, axis=1, keepdims=True)

    # Avoid division by zero
    safe = (norms_a[:, 0] > 0) & (norms_b[:, 0] > 0)
    result = np.full(len(vecs_a), np.nan)
    if safe.any():
        a_norm = vecs_a[safe] / norms_a[safe]
        b_norm = vecs_b[safe] / norms_b[safe]
        result[safe] = (a_norm * b_norm).sum(axis=1)
    return result


def js_divergence_topk(
    topk_a: List[List[dict]],  # [positions][top_k items] each item: {token, prob}
    topk_b: List[List[dict]],
) -> float:
    """Average Jensen-Shannon divergence of top-k distributions across positions.

    Lower values indicate more similar distributions.
    Returns average JS divergence in [0, ln2].
    """
    js_values = []
    for pos_a, pos_b in zip(topk_a, topk_b):
        # Build unified token vocabulary for this position
        tokens_a = {item["token"]: item["prob"] for item in pos_a}
        tokens_b = {item["token"]: item["prob"] for item in pos_b}
        all_tokens = list(set(tokens_a) | set(tokens_b))

        p = np.array([tokens_a.get(t, 0.0) for t in all_tokens], dtype=np.float64)
        q = np.array([tokens_b.get(t, 0.0) for t in all_tokens], dtype=np.float64)

        # Renormalize (top_k probs may not sum to 1)
        p_sum, q_sum = p.sum(), q.sum()
        if p_sum == 0 or q_sum == 0:
            continue
        p /= p_sum
        q /= q_sum

        m = (p + q) / 2
        # Use log with clip to avoid log(0)
        def kl(a: np.ndarray, b: np.ndarray) -> float:
            mask = a > 0
            return float(np.sum(a[mask] * np.log(a[mask] / np.clip(b[mask], 1e-12, None))))

        js = (kl(p, m) + kl(q, m)) / 2
        js_values.append(js)

    return float(np.mean(js_values)) if js_values else float("nan")


def token_agreement_rate(
    tokens_a: List[str],
    tokens_b: List[str],
) -> float:
    """Fraction of positions where both models chose the same top-1 token."""
    if not tokens_a or not tokens_b:
        return float("nan")
    n = min(len(tokens_a), len(tokens_b))
    matches = sum(1 for a, b in zip(tokens_a[:n], tokens_b[:n]) if a == b)
    return matches / n


def all_pairs_cosine(
    miners: List[Tuple[int, np.ndarray]],  # list of (uid, task_matrix (n_tasks, n_tokens))
    common_task_ids: List[int],
) -> Dict[Tuple[int, int], np.ndarray]:
    """Vectorized all-pairs cosine similarity.

    For each task, stack all miners into a matrix and compute the full
    similarity matrix in one shot. Returns {(uid_a, uid_b): cosine_array}.
    """
    n = len(miners)
    uids = [uid for uid, _ in miners]
    # matrices: list of (n_tasks, n_tokens) per miner
    matrices = [mat for _, mat in miners]  # each (n_tasks, n_tokens)

    # Stack into (n_miners, n_tasks, n_tokens)
    stacked = np.stack(matrices, axis=0)  # (n, n_tasks, n_tokens)

    # Normalize along token dimension
    norms = np.linalg.norm(stacked, axis=2, keepdims=True)  # (n, n_tasks, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        normed = np.where(norms > 0, stacked / norms, 0.0)  # (n, n_tasks, n_tokens)

    # Per-task similarity matrix: (n_tasks, n, n)
    # normed[:, t, :] has shape (n, n_tokens)
    # sim[t] = normed[:,t,:] @ normed[:,t,:].T
    result: Dict[Tuple[int, int], np.ndarray] = {}

    n_tasks = stacked.shape[1]
    # Build (n_tasks, n, n) similarity cube
    sim_cube = np.einsum("itk,jtk->tij", normed, normed)  # (n_tasks, n, n)

    for i in range(n):
        for j in range(i + 1, n):
            result[(uids[i], uids[j])] = sim_cube[:, i, j]  # (n_tasks,)

    return result

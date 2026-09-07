from typing import Dict, Iterable, Sequence


def recall_at_k_from_sim(sim: Sequence[Sequence[float]], ks: Iterable[int] = (1, 5, 10)) -> Dict[int, float]:
    rows = [list(r) for r in sim]
    if not rows:
        raise ValueError("sim must be non-empty")
    n = len(rows)
    if any(len(r) != n for r in rows):
        shape = (len(rows), len(rows[0]) if rows else 0)
        raise ValueError(f"sim must be square [N, N], got {shape}")

    out: Dict[int, float] = {}
    for k in ks:
        if int(k) <= 0:
            raise ValueError(f"k must be positive, got {k}")
        k_eff = min(int(k), n)
        hits = 0
        for i, row in enumerate(rows):
            top_idx = sorted(range(n), key=lambda j: row[j], reverse=True)[:k_eff]
            if i in top_idx:
                hits += 1
        out[int(k)] = hits / n
    return out


def recall_at_k_multi_positive(
    sim: Sequence[Sequence[float]],
    positive_indices: Sequence[Iterable[int]],
    ks: Iterable[int] = (1, 5, 10),
) -> Dict[int, float]:
    """Compute Recall@K for rectangular similarities with one or more positives.

    ``positive_indices[i]`` contains the candidate indices that are correct for
    query row ``i``. Queries without a positive candidate are rejected because
    silently counting them as misses makes split/manifest errors hard to spot.
    """
    rows = [list(row) for row in sim]
    if not rows:
        raise ValueError("sim must be non-empty")
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError("sim must be a non-empty rectangular matrix")
    if len(positive_indices) != len(rows):
        raise ValueError("positive_indices must contain one entry per query")

    positives: list[set[int]] = []
    for query_idx, values in enumerate(positive_indices):
        current = {int(value) for value in values}
        if not current:
            raise ValueError(f"query {query_idx} has no positive candidate")
        if min(current) < 0 or max(current) >= width:
            raise ValueError(f"query {query_idx} contains an out-of-range positive index")
        positives.append(current)

    ranked = [sorted(range(width), key=lambda idx: row[idx], reverse=True) for row in rows]
    out: Dict[int, float] = {}
    for k in ks:
        if int(k) <= 0:
            raise ValueError(f"k must be positive, got {k}")
        k_eff = min(int(k), width)
        hits = sum(bool(set(order[:k_eff]) & positive) for order, positive in zip(ranked, positives))
        out[int(k)] = hits / len(rows)
    return out


def mean_reciprocal_rank_multi_positive(
    sim: Sequence[Sequence[float]],
    positive_indices: Sequence[Iterable[int]],
) -> float:
    """Return MRR using the rank of the first relevant candidate per query."""
    rows = [list(row) for row in sim]
    positives = [{int(value) for value in values} for values in positive_indices]
    # Reuse validation, including the no-positive invariant.
    recall_at_k_multi_positive(rows, positives, ks=(1,))
    reciprocal_ranks = []
    for row, relevant in zip(rows, positives):
        order = sorted(range(len(row)), key=lambda idx: row[idx], reverse=True)
        rank = next(rank for rank, idx in enumerate(order, start=1) if idx in relevant)
        reciprocal_ranks.append(1.0 / rank)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)

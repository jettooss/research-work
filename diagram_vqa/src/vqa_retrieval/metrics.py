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

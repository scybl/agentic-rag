"""确定性 RRF 和基于余弦冗余的 MMR；保留每个阶段的排名。"""

import math

import numpy as np


def rrf(rankings, *, constant=60):
    if constant < 1:
        raise ValueError("RRF 常数必须为正")
    scores = {}
    for ranking in rankings:
        seen = set()
        for rank, (document_id, _) in enumerate(ranking, 1):
            if document_id in seen:
                raise ValueError("同一路召回不能重复出现材料")
            seen.add(document_id)
            scores[document_id] = scores.get(document_id, 0.0) + 1.0 / (constant + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


def mmr_order(ranking, vectors, *, top_k, weight=0.7):
    if not math.isfinite(weight) or not 0 <= weight <= 1 or top_k < 1:
        raise ValueError("MMR 权重须在 0–1，上限必须为正")
    ids = [key for key, _ in ranking]
    if len(set(ids)) != len(ids):
        raise ValueError("MMR 输入存在重复材料")
    if not ids:
        return []
    # 各阶段原始分数尺度不同，使用明确的线性排名相关度，不能直接混加 BM25 与 logits。
    relevance = {key: 1.0 - rank / max(1, len(ids) - 1) for rank, key in enumerate(ids)}
    chosen, remaining = [], list(ids)
    while remaining and len(chosen) < top_k:
        def utility(key):
            redundancy = max((float(np.dot(vectors[key], vectors[c])) for c in chosen), default=0.0)
            return weight * relevance[key] - (1 - weight) * max(0.0, min(1.0, redundancy))
        best = min(remaining, key=lambda key: (-utility(key), ids.index(key), key))
        chosen.append(best)
        remaining.remove(best)
    return chosen + remaining

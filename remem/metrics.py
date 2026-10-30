"""Dependency-free metrics; scores and labels are aligned per candidate."""
import math
from collections import defaultdict


def binary_auc(labels, scores):
    positives, negatives = sum(labels), len(labels) - sum(labels)
    if positives == 0 or negatives == 0:
        return None
    ordered = sorted(zip(scores, labels))
    rank_sum, i = 0.0, 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        mean_rank = (i + 1 + j) / 2
        rank_sum += mean_rank * sum(y for _, y in ordered[i:j])
        i = j
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def evaluate_metrics(labels, scores, groups, k=10):
    if not labels or len(labels) != len(scores) or len(scores) != len(groups):
        raise ValueError("metric inputs must have the same nonzero length")
    eps = 1e-7
    loss = -sum(y * math.log(max(eps, min(1-eps, p))) + (1-y) * math.log(max(eps, min(1-eps, 1-p)))
                for y, p in zip(labels, scores)) / len(labels)
    grouped = defaultdict(list)
    for y, p, g in zip(labels, scores, groups):
        grouped[g].append((p, y))
    ndcgs, recalls = [], []
    skipped = 0
    for values in grouped.values():
        positives = sum(y for _, y in values)
        # No-positive or singleton groups do not constitute informative ranking tests.
        if positives == 0 or len(values) < 2:
            skipped += 1
            continue
        ranked = sorted(values, key=lambda v: -v[0])[:k]  # stable ties
        dcg = sum(y / math.log2(i+2) for i, (_, y) in enumerate(ranked))
        ideal = sum(1 / math.log2(i+2) for i in range(min(k, positives)))
        ndcgs.append(dcg / ideal)
        recalls.append(sum(y for _, y in ranked) / positives)
    return {"auc": binary_auc(labels, scores), "logloss": loss,
            f"ndcg@{k}": sum(ndcgs)/len(ndcgs) if ndcgs else None,
            f"recall@{k}": sum(recalls)/len(recalls) if recalls else None,
            "ranking_groups": len(ndcgs), "skipped_groups": skipped}

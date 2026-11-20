from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from app.data import Session
from app.vocab import Vocabulary


def product_frequency(sessions: Iterable[Session]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for session in sessions:
        counts.update(session.items)
    return counts


def rare_product_summary(sessions: list[Session], vocabulary: Vocabulary, evaluations: dict[str, dict[str, object]], threshold: int = 10) -> list[dict[str, object]]:
    counts = product_frequency(sessions)
    rows = []
    for product_id in vocabulary.tokens[2:]:
        count = counts.get(product_id, 0)
        if count < threshold:
            rows.append({
                "product": product_id,
                "test_frequency": count,
                "model_ranks": {
                    model: evaluation["per_product_mean_rank"].get(product_id)
                    for model, evaluation in evaluations.items()
                },
            })
    return sorted(rows, key=lambda row: (row["test_frequency"], row["product"]))

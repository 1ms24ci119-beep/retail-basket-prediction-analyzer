from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

BOS = -1


class NGramModel:
    def __init__(self, vocab_size: int, product_ids: Iterable[int], order: int = 3, alpha: float = 0.2):
        if order < 1 or order > 4:
            raise ValueError("order must be between 1 and 4")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.vocab_size = int(vocab_size)
        self.product_ids = tuple(sorted({int(item) for item in product_ids}))
        if self.vocab_size <= 0 or not self.product_ids or any(i < 0 or i >= self.vocab_size for i in self.product_ids):
            raise ValueError("product ids must be non-empty indices inside the vocabulary")
        self.order = order
        self.alpha = float(alpha)
        self.counts: dict[int, dict[tuple[int, ...], Counter[int]]] = {
            n: defaultdict(Counter) for n in range(1, order + 1)
        }
        self.fitted = False

    def fit(self, sequences: Iterable[Sequence[int]]) -> NGramModel:
        seen = 0
        for sequence in sequences:
            products = [int(item) for item in sequence]
            if any(item not in self.product_ids for item in products):
                raise ValueError("training sequences may contain only product token ids")
            if not products:
                continue
            seen += 1
            history = [BOS] * (self.order - 1)
            for token in products:
                for context_length in range(self.order):
                    n = context_length + 1
                    context = tuple(history[-context_length:]) if context_length else ()
                    self.counts[n][context][token] += 1
                history.append(token)
        if not seen:
            raise ValueError("at least one non-empty sequence is required")
        self.fitted = True
        return self

    def _order_distribution(self, context: tuple[int, ...], context_length: int) -> dict[int, float] | None:
        counts = self.counts[context_length + 1].get(context)
        if not counts:
            return None
        denominator = sum(counts.values()) + self.alpha * len(self.product_ids)
        return {item: (counts.get(item, 0) + self.alpha) / denominator for item in self.product_ids}

    def predict_proba(self, prefix: Sequence[int]) -> dict[int, float]:
        if not self.fitted:
            raise RuntimeError("fit the model before requesting predictions")
        values = [int(item) for item in prefix]
        if any(item not in self.product_ids for item in values):
            raise ValueError("prefix may contain only product token ids")
        available = self.order - 1 if not values else min(len(values), self.order - 1)
        context = tuple(values[-available:]) if values else (BOS,) * available
        distributions = []
        for context_length in range(available + 1):
            suffix = tuple(context[-context_length:]) if context_length else ()
            distribution = self._order_distribution(suffix, context_length)
            if distribution is not None:
                weight = 0.55 ** (available - context_length)
                distributions.append((weight, distribution))
        if not distributions:
            raise RuntimeError("unigram counts are unavailable")
        total_weight = sum(weight for weight, _ in distributions)
        result = {
            item: sum(weight * distribution[item] for weight, distribution in distributions) / total_weight
            for item in self.product_ids
        }
        normalizer = sum(result.values())
        return {item: probability / normalizer for item, probability in result.items()}

    def predict(self, prefix: Sequence[int], top_k: int = 5) -> list[tuple[int, float]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        probabilities = self.predict_proba(prefix)
        return sorted(probabilities.items(), key=lambda pair: (-pair[1], pair[0]))[:top_k]

    def to_dict(self) -> dict[str, object]:
        return {
            "vocab_size": self.vocab_size,
            "product_ids": list(self.product_ids),
            "order": self.order,
            "alpha": self.alpha,
            "counts": {
                str(n): [
                    {"context": list(context), "next": dict(next_counts)}
                    for context, next_counts in contexts.items()
                ]
                for n, contexts in self.counts.items()
            },
            "fitted": self.fitted,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> NGramModel:
        model = cls(int(value["vocab_size"]), value["product_ids"], int(value["order"]), float(value["alpha"]))
        for order, contexts in value["counts"].items():
            for row in contexts:
                context = tuple(int(item) for item in row["context"])
                next_counts = Counter({int(item): int(count) for item, count in row["next"].items()})
                model.counts[int(order)][context] = next_counts
        model.fitted = bool(value["fitted"])
        return model

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), separators=(",", ":")), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> NGramModel:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

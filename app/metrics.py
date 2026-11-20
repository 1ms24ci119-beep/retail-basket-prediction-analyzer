from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Callable, Iterable

from app.data import Session, product_lookup
from app.vocab import Vocabulary


Predictor = Callable[[list[int]], dict[int, float]]


def prediction_rows(sessions: Iterable[Session], vocabulary: Vocabulary):
    for session in sessions:
        encoded = vocabulary.encode(session.items)
        for target_index in range(1, len(encoded)):
            yield encoded[:target_index], encoded[target_index], session


def evaluate_predictor(
    predictor: Predictor,
    sessions: list[Session],
    vocabulary: Vocabulary,
    top_k_values: tuple[int, ...] = (1, 3, 5),
) -> dict[str, object]:
    if not sessions:
        raise ValueError("evaluation split is empty")
    if not top_k_values or any(k < 1 for k in top_k_values):
        raise ValueError("top-k cutoffs must be positive")
    ids = vocabulary.product_ids
    targets: list[int] = []
    ranks: list[int] = []
    cross_entropies: list[float] = []
    distributions: list[dict[int, float]] = []
    contexts: list[list[int]] = []
    sessions_for_row: list[Session] = []
    for prefix, target, session in prediction_rows(sessions, vocabulary):
        probabilities = predictor(prefix)
        if set(probabilities) != set(ids):
            raise ValueError("predictor must return probabilities for the full product vocabulary")
        total = sum(probabilities.values())
        if any(probability < 0 or not math.isfinite(probability) for probability in probabilities.values()):
            raise ValueError("prediction probabilities must be finite and non-negative")
        if not math.isclose(total, 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError("prediction probabilities must sum to one")
        ranked = sorted(ids, key=lambda item: (-probabilities[item], item))
        targets.append(target)
        ranks.append(ranked.index(target) + 1)
        cross_entropies.append(-math.log(max(probabilities[target], 1e-15)))
        distributions.append(probabilities)
        contexts.append(prefix)
        sessions_for_row.append(session)
    count = len(targets)
    metrics: dict[str, object] = {
        "examples": count,
        "top_1_accuracy": sum(rank <= 1 for rank in ranks) / count,
        "top_3_accuracy": sum(rank <= 3 for rank in ranks) / count,
        "top_5_accuracy": sum(rank <= 5 for rank in ranks) / count,
        "mrr": sum(1 / rank for rank in ranks) / count,
        "mean_rank": sum(ranks) / count,
        "cross_entropy": sum(cross_entropies) / count,
    }
    catalogue = product_lookup()
    category_hits: dict[str, list[bool]] = defaultdict(list)
    size_hits: dict[str, list[bool]] = defaultdict(list)
    item_ranks: dict[str, list[int]] = defaultdict(list)
    for target, rank, prefix in zip(targets, ranks, contexts):
        name = vocabulary.tokens[target]
        category_hits[catalogue[name].category].append(rank == 1)
        bucket = "1-2" if len(prefix) <= 2 else "3-4" if len(prefix) <= 4 else "5+"
        size_hits[bucket].append(rank == 1)
        item_ranks[name].append(rank)
    metrics["per_category_accuracy"] = {
        category: sum(hits) / len(hits) for category, hits in sorted(category_hits.items())
    }
    metrics["accuracy_by_prefix_size"] = {
        bucket: sum(hits) / len(hits) for bucket, hits in sorted(size_hits.items())
    }
    metrics["per_product_mean_rank"] = {
        name: sum(values) / len(values) for name, values in sorted(item_ranks.items())
    }
    metrics["_rows"] = [
        {"prefix": prefix, "target": target, "rank": rank, "probabilities": distribution, "session": session}
        for prefix, target, rank, distribution, session in zip(contexts, targets, ranks, distributions, sessions_for_row)
    ]
    return metrics


def summarize_errors(results: dict[str, dict[str, object]], vocabulary: Vocabulary, limit: int = 20) -> dict[str, object]:
    if not {"ngram", "transformer"}.issubset(results):
        raise ValueError("results must contain ngram and transformer evaluations")
    ngram_rows = results["ngram"]["_rows"]
    transformer_rows = results["transformer"]["_rows"]
    if len(ngram_rows) != len(transformer_rows):
        raise ValueError("models must be evaluated on the same examples")
    by_item: dict[str, Counter[str]] = defaultdict(Counter)
    cases = []
    ngram_better = 0
    transformer_better = 0
    for ngram_row, transformer_row in zip(ngram_rows, transformer_rows):
        if ngram_row["target"] != transformer_row["target"] or ngram_row["prefix"] != transformer_row["prefix"]:
            raise ValueError("model evaluation rows are not aligned")
        target = vocabulary.tokens[ngram_row["target"]]
        ngram_guess = max(ngram_row["probabilities"], key=ngram_row["probabilities"].get)
        transformer_guess = max(transformer_row["probabilities"], key=transformer_row["probabilities"].get)
        by_item[target][f"ngram_rank_{ngram_row['rank']}"] += 1
        if ngram_row["rank"] < transformer_row["rank"]:
            ngram_better += 1
        elif transformer_row["rank"] < ngram_row["rank"]:
            transformer_better += 1
        if ngram_row["rank"] != transformer_row["rank"]:
            cases.append({
                "session_id": ngram_row["session"].session_id,
                "prefix": [vocabulary.tokens[item] for item in ngram_row["prefix"]],
                "actual": target,
                "ngram_top1": vocabulary.tokens[ngram_guess],
                "ngram_rank": ngram_row["rank"],
                "transformer_top1": vocabulary.tokens[transformer_guess],
                "transformer_rank": transformer_row["rank"],
            })
    confusions = Counter()
    for model_name, evaluation in results.items():
        for row in evaluation["_rows"]:
            guess = max(row["probabilities"], key=row["probabilities"].get)
            if guess != row["target"]:
                confusions[(vocabulary.tokens[row["target"]], vocabulary.tokens[guess])] += 1
    return {
        "ngram_better_examples": ngram_better,
        "transformer_better_examples": transformer_better,
        "different_rank_examples": len(cases),
        "top_confusions": [
            {"actual": actual, "predicted": predicted, "count": count}
            for (actual, predicted), count in confusions.most_common(limit)
        ],
        "representative_cases": cases[:limit],
        "rank_by_product": {key: dict(value) for key, value in sorted(by_item.items())},
    }


def public_metrics(result: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in result.items() if not key.startswith("_")}

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from app.data import Session, describe_sessions, generate_sessions, save_sessions, split_sessions
from app.metrics import evaluate_predictor, public_metrics, summarize_errors
from app.ngram import NGramModel
from app.training import choose_training_subset, fit_ngram, fit_transformer
from app.transformer import TransformerConfig
from app.vocab import Vocabulary


DEFAULT_DATA = Path("data/generated")
DEFAULT_ARTIFACTS = Path("artifacts")
DEFAULT_RESULTS = Path("results")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _comparison_rows(evaluations: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    metrics = ("top_1_accuracy", "top_3_accuracy", "top_5_accuracy", "mrr", "mean_rank", "cross_entropy")
    return [{"metric": metric, **{model: values[metric] for model, values in evaluations.items()}} for metric in metrics]


def train_and_evaluate(
    splits: dict[str, list[Session]],
    vocabulary: Vocabulary,
    artifacts_dir: Path,
    fraction: float = 1.0,
    epochs: int = 6,
    seed: int = 1729,
    batch_size: int = 64,
    device: str = "auto",
    save_models: bool = True,
) -> tuple[dict[str, dict[str, object]], dict[str, object], dict[str, list[float]]]:
    train_subset = choose_training_subset(splits["train"], fraction, seed)
    artifact_label = f"scale_{fraction:.2f}".replace(".", "_")
    ngram_path = artifacts_dir / f"ngram_{artifact_label}.json"
    transformer_path = artifacts_dir / f"transformer_{artifact_label}.pt"
    ngram, ngram_metadata = fit_ngram(train_subset, vocabulary, ngram_path if save_models else Path("/tmp/retail-ngram.json"))
    model, history, transformer_metadata = fit_transformer(
        train_subset, splits["validation"], vocabulary,
        transformer_path if save_models else Path("/tmp/retail-transformer.pt"),
        epochs=epochs, batch_size=batch_size, seed=seed, device=device,
    )
    model_device = next(model.parameters()).device
    ngram_eval = evaluate_predictor(ngram.predict_proba, splits["test"], vocabulary)
    transformer_eval = evaluate_predictor(
        lambda prefix: model.predict_proba(prefix, vocabulary.product_ids, model_device),
        splits["test"], vocabulary,
    )
    evaluations = {"ngram": ngram_eval, "transformer": transformer_eval}
    analysis = summarize_errors(evaluations, vocabulary)
    metadata = {
        "training_fraction": fraction,
        "ngram": ngram_metadata,
        "transformer": transformer_metadata,
        "parameter_comparison": {
            "ngram_contexts": sum(len(contexts) for contexts in ngram.counts.values()),
            "transformer_parameters": transformer_metadata["parameters"],
        },
    }
    return evaluations, {"error_analysis": analysis, "metadata": metadata}, history


def run_experiment(
    count: int = 1200,
    seed: int = 1729,
    epochs: int = 6,
    fractions: Iterable[float] = (1.0,),
    data_dir: Path = DEFAULT_DATA,
    artifacts_dir: Path = DEFAULT_ARTIFACTS,
    results_dir: Path = DEFAULT_RESULTS,
    batch_size: int = 64,
    device: str = "auto",
) -> dict[str, object]:
    sessions = generate_sessions(count=count, seed=seed)
    splits = split_sessions(sessions, seed=seed)
    save_sessions(splits, data_dir)
    vocabulary = Vocabulary.from_catalogue()
    (artifacts_dir / "vocabulary.json").parent.mkdir(parents=True, exist_ok=True)
    write_json(artifacts_dir / "vocabulary.json", vocabulary.to_dict())
    stats = {split_name: describe_sessions(rows) for split_name, rows in splits.items()}
    stats["split_sizes"] = {name: len(rows) for name, rows in splits.items()}
    write_json(results_dir / "dataset_stats.json", stats)
    pd.DataFrame([{"split": split_name, **value} for split_name, value in stats.items() if split_name in splits]).to_csv(
        results_dir / "dataset_split_summary.csv", index=False
    )
    all_scale_results: dict[str, object] = {}
    final_history: dict[str, list[float]] = {}
    for fraction in fractions:
        if not 0 < fraction <= 1:
            raise ValueError("training fractions must be in (0, 1]")
        evaluations, auxiliary, history = train_and_evaluate(
            splits, vocabulary, artifacts_dir, fraction=fraction, epochs=epochs,
            seed=seed, batch_size=batch_size, device=device,
        )
        key = f"{fraction:.2f}"
        all_scale_results[key] = {
            "metrics": {name: public_metrics(result) for name, result in evaluations.items()},
            **auxiliary,
        }
        final_history[key] = history
    report = {
        "seed": seed,
        "transaction_count": count,
        "split_sizes": stats["split_sizes"],
        "vocabulary_size": len(vocabulary.tokens),
        "sequence_convention": "Template order for planted sessions, random order for independent and unusual sessions; all session IDs are split before prefixes are evaluated.",
        "training_sizes": all_scale_results,
        "limitations": [
            "Transactions are synthetic and generated from known templates.",
            "Within-template item order is simulated and does not represent observed checkout timestamps.",
            "Randomized sessions can make next-item prediction intrinsically uncertain.",
        ],
    }
    write_json(results_dir / "metrics.json", report)
    write_json(results_dir / "training_history.json", final_history)
    table_rows = []
    for fraction, result in all_scale_results.items():
        for row in _comparison_rows(result["metrics"]):
            table_rows.append({"training_fraction": float(fraction), **row})
    pd.DataFrame(table_rows).to_csv(results_dir / "model_comparison.csv", index=False)
    return report


def load_report(path: Path = DEFAULT_RESULTS / "metrics.json") -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"No evaluation report found at {path}; run `python -m app.cli run-all` first")
    return json.loads(path.read_text(encoding="utf-8"))

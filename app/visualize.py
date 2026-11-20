from __future__ import annotations

import json
import os
from pathlib import Path

cache_root = Path(".cache").resolve()
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from app.experiment import DEFAULT_ARTIFACTS, DEFAULT_RESULTS

COLORS = {"ngram": "#31688e", "transformer": "#d1495b"}


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run the experiment first")
    return json.loads(path.read_text(encoding="utf-8"))


def create_visualizations(
    results_dir: Path = DEFAULT_RESULTS, artifacts_dir: Path = DEFAULT_ARTIFACTS
) -> list[Path]:
    report = _load(results_dir / "metrics.json")
    history = _load(results_dir / "training_history.json")
    stats = _load(results_dir / "dataset_stats.json")
    output_paths: list[Path] = []
    results_dir.mkdir(parents=True, exist_ok=True)

    fractions = sorted(report["training_sizes"], key=float)
    metric_keys = ["top_1_accuracy", "top_3_accuracy", "top_5_accuracy", "mrr"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True, layout="constrained")
    for axis, metric in zip(axes.flat, metric_keys):
        for model in ("ngram", "transformer"):
            values = [report["training_sizes"][fraction]["metrics"][model][metric] for fraction in fractions]
            axis.plot([float(value) for value in fractions], values, marker="o", linewidth=2,
                      color=COLORS[model], label=model.replace("_", " ").title())
        display_name = {
            "top_1_accuracy": "Top-1 accuracy",
            "top_3_accuracy": "Top-3 accuracy",
            "top_5_accuracy": "Top-5 accuracy",
            "mrr": "Mean reciprocal rank",
        }[metric]
        axis.set_title(display_name)
        axis.set_ylim(0, 1)
        axis.set_xlabel("Share of training transactions")
        axis.set_xticks([float(value) for value in fractions], [f"{float(value):.0%}" for value in fractions])
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].set_ylabel("Score")
    axes[1, 0].set_ylabel("Score")
    axes[0, 0].legend()
    fig.suptitle("Next-item ranking scores by model and training size")
    path = results_dir / "model_performance.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    output_paths.append(path)

    fraction_for_history = fractions[-1]
    losses = history[fraction_for_history]
    fig, ax = plt.subplots(figsize=(7.5, 4.6), layout="constrained")
    epochs = range(1, len(losses["train_loss"]) + 1)
    ax.plot(epochs, losses["train_loss"], marker="o", label="Training")
    ax.plot(epochs, losses["validation_loss"], marker="o", label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Next-token cross-entropy")
    ax.set_title(f"Transformer learning curves ({float(fraction_for_history):.0%} training data)")
    ax.legend()
    path = results_dir / "training_curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    output_paths.append(path)

    frequencies = stats["train"]["product_frequency"]
    most_common = sorted(frequencies.items(), key=lambda pair: pair[1], reverse=True)[:20]
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    labels, values = zip(*most_common) if most_common else ([], [])
    ax.barh(list(reversed(labels)), list(reversed(values)), color="#4f772d")
    ax.set_xlabel("Training session count")
    ax.set_title("Most frequent products in the training split")
    path = results_dir / "product_frequency.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    output_paths.append(path)

    lengths = stats["train"]["basket_length_distribution"]
    fig, ax = plt.subplots(figsize=(7.5, 4.6), layout="constrained")
    ax.bar([int(key) for key in lengths], list(lengths.values()), color="#f4a261")
    ax.set_xlabel("Products in session")
    ax.set_ylabel("Number of training sessions")
    ax.set_title("Training session length distribution")
    path = results_dir / "basket_lengths.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    output_paths.append(path)

    confusion = report["training_sizes"][fraction_for_history]["error_analysis"]["top_confusions"][:12]
    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    if confusion:
        labels = [f"{row['actual']} → {row['predicted']}" for row in confusion]
        ax.barh(list(reversed(labels)), [row["count"] for row in reversed(confusion)], color="#8d5a97")
    ax.set_xlabel("Wrong top-1 predictions")
    ax.set_title("Most common product confusions across both models")
    path = results_dir / "prediction_confusions.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    output_paths.append(path)
    output_paths.append(create_prediction_figure(results_dir, artifacts_dir, float(fraction_for_history)))
    return output_paths


def create_prediction_figure(results_dir: Path, artifacts_dir: Path, fraction: float = 1.0) -> Path:
    from app.data import product_lookup
    from app.ngram import NGramModel
    from app.transformer import load_checkpoint

    label = f"scale_{fraction:.2f}".replace(".", "_")
    ngram_path = artifacts_dir / f"ngram_{label}.json"
    transformer_path = artifacts_dir / f"transformer_{label}.pt"
    if not ngram_path.exists() or not transformer_path.exists():
        raise FileNotFoundError(f"trained models for {fraction:.0%} data are needed to draw a prediction example")
    ngram = NGramModel.load(ngram_path)
    transformer, vocabulary, _ = load_checkpoint(transformer_path)
    basket = ["bakery_bread", "dairy_butter"]
    prefix = vocabulary.encode(basket)
    transformer_device = next(transformer.parameters()).device
    predictions = {
        "N-gram": ngram.predict_proba(prefix),
        "Transformer": transformer.predict_proba(prefix, vocabulary.product_ids, transformer_device),
    }
    catalogue = product_lookup()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")
    for axis, (model_name, distribution) in zip(axes, predictions.items()):
        top = sorted(distribution.items(), key=lambda pair: (-pair[1], pair[0]))[:5]
        labels = [catalogue[vocabulary.tokens[item]].name for item, _ in top]
        values = [probability for _, probability in top]
        color_key = "ngram" if model_name == "N-gram" else "transformer"
        axis.barh(list(reversed(labels)), list(reversed(values)), color=COLORS[color_key])
        axis.set_title(model_name)
        axis.set_xlabel("Probability within product vocabulary")
        axis.set_xlim(0, max(values) * 1.2 if values else 1)
    fig.suptitle("Next-item suggestions after Bread and Butter")
    path = results_dir / "prediction_example.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def comparison_table(results_dir: Path = DEFAULT_RESULTS) -> pd.DataFrame:
    path = results_dir / "model_comparison.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run an evaluation first")
    return pd.read_csv(path)

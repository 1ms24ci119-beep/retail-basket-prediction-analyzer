"""Command-line workflows for generating and comparing retail next-item models."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from app.config import BATCH_SIZE, DEVICE, EPOCHS, SEED, TRANSACTIONS
from app.data import (
    CATALOGUE,
    describe_sessions,
    generate_sessions,
    load_sessions,
    save_sessions,
    split_sessions,
)
from app.experiment import (
    DEFAULT_ARTIFACTS,
    DEFAULT_DATA,
    DEFAULT_RESULTS,
    run_experiment,
    write_json,
)
from app.metrics import evaluate_predictor, public_metrics, summarize_errors
from app.training import (
    choose_training_subset,
    fit_ngram,
    fit_transformer,
    restore_models,
)
from app.transformer import TransformerConfig, choose_device
from app.visualize import comparison_table, create_visualizations
from app.vocab import Vocabulary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="retail-basket-prediction-analyzer",
        description="Compare a from-scratch n-gram model with a small causal Transformer on synthetic purchase sessions.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    hello = subparsers.add_parser("hello", help="check that the starter interfaces still work")
    hello.set_defaults(handler=_hello)

    generate = subparsers.add_parser("generate-data", help="generate reproducible train, validation, and test sessions")
    generate.add_argument("--transactions", type=int, default=TRANSACTIONS)
    generate.add_argument("--seed", type=int, default=SEED)
    generate.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    generate.set_defaults(handler=_generate)

    stats = subparsers.add_parser("stats", help="summarize the saved dataset splits")
    stats.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    stats.set_defaults(handler=_stats)

    train_ngram = subparsers.add_parser("train-ngram", help="fit and save the statistical model")
    _add_train_options(train_ngram)
    train_ngram.add_argument("--order", type=int, choices=range(1, 5), default=3)
    train_ngram.add_argument("--alpha", type=float, default=0.2)
    train_ngram.set_defaults(handler=_train_ngram)

    train_transformer = subparsers.add_parser("train-transformer", help="fit and save the small PyTorch model")
    _add_train_options(train_transformer)
    train_transformer.add_argument("--epochs", type=int, default=EPOCHS)
    train_transformer.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    train_transformer.add_argument("--device", default=DEVICE)
    train_transformer.add_argument("--d-model", type=int, default=64)
    train_transformer.add_argument("--n-heads", type=int, default=4)
    train_transformer.add_argument("--n-layers", type=int, default=2)
    train_transformer.add_argument("--block-size", type=int, default=12)
    train_transformer.add_argument("--dropout", type=float, default=0.1)
    train_transformer.add_argument("--learning-rate", type=float, default=0.002)
    train_transformer.set_defaults(handler=_train_transformer)

    evaluate = subparsers.add_parser("evaluate", help="evaluate saved models against the common test split")
    _add_path_options(evaluate)
    evaluate.add_argument("--fraction", type=float, default=1.0)
    evaluate.add_argument("--device", default=DEVICE)
    evaluate.set_defaults(handler=_evaluate)

    compare = subparsers.add_parser("compare", help="print model metrics from the latest experiment")
    compare.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    compare.set_defaults(handler=_compare)

    predict = subparsers.add_parser("predict", help="rank next products for a comma-separated basket")
    predict.add_argument("--basket", required=True, help="product IDs or names, for example bread,butter,milk")
    predict.add_argument("--top-k", type=int, default=5)
    predict.add_argument("--fraction", type=float, default=1.0)
    predict.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    predict.add_argument("--device", default="cpu")
    predict.set_defaults(handler=_predict)

    run_all = subparsers.add_parser("run-all", help="generate data, train models, evaluate, and create plots")
    run_all.add_argument("--transactions", type=int, default=TRANSACTIONS)
    run_all.add_argument("--seed", type=int, default=SEED)
    run_all.add_argument("--epochs", type=int, default=EPOCHS)
    run_all.add_argument("--fractions", type=float, nargs="+", default=[1.0])
    run_all.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    run_all.add_argument("--device", default=DEVICE)
    run_all.add_argument("--d-model", type=int, default=64)
    run_all.add_argument("--n-heads", type=int, default=4)
    run_all.add_argument("--n-layers", type=int, default=2)
    run_all.add_argument("--block-size", type=int, default=12)
    run_all.add_argument("--dropout", type=float, default=0.1)
    run_all.add_argument("--learning-rate", type=float, default=0.002)
    run_all.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    run_all.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    run_all.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    run_all.set_defaults(handler=_run_all)

    visualize = subparsers.add_parser("visualize", help="create figures from the latest saved experiment")
    visualize.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    visualize.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    visualize.set_defaults(handler=_visualize)
    return parser


def _scale_label(fraction: float) -> str:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    return f"scale_{fraction:.2f}".replace(".", "_")


def _add_path_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)


def _add_train_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=SEED)


def _load_splits(data_dir: Path):
    expected = [data_dir / f"{name}.jsonl" for name in ("train", "validation", "test")]
    if not all(path.exists() for path in expected):
        raise FileNotFoundError("dataset splits are missing; run `python -m app.cli generate-data` first")
    return {name: load_sessions(data_dir / f"{name}.jsonl") for name in ("train", "validation", "test")}


def _hello(args: argparse.Namespace) -> int:
    print("Retail Basket Prediction Analyzer is ready. Run `python -m app.cli --help` to see the experiment commands.")
    return 0


def _generate(args: argparse.Namespace) -> int:
    splits = split_sessions(generate_sessions(args.transactions, args.seed), seed=args.seed)
    save_sessions(splits, args.data_dir)
    print(json.dumps({name: len(rows) for name, rows in splits.items()}, indent=2))
    return 0


def _stats(args: argparse.Namespace) -> int:
    splits = _load_splits(args.data_dir)
    summary = {name: describe_sessions(rows) for name, rows in splits.items()}
    print(json.dumps(summary, indent=2))
    return 0


def _train_ngram(args: argparse.Namespace) -> int:
    splits = _load_splits(args.data_dir)
    vocabulary = Vocabulary.from_catalogue()
    subset = choose_training_subset(splits["train"], args.fraction, args.seed)
    label = _scale_label(args.fraction)
    model, metadata = fit_ngram(subset, vocabulary, args.artifacts_dir / f"ngram_{label}.json", args.order, args.alpha)
    write_json(args.artifacts_dir / "vocabulary.json", vocabulary.to_dict())
    print(json.dumps({**metadata, "contexts": sum(len(value) for value in model.counts.values())}, indent=2))
    return 0


def _train_transformer(args: argparse.Namespace) -> int:
    splits = _load_splits(args.data_dir)
    vocabulary = Vocabulary.from_catalogue()
    subset = choose_training_subset(splits["train"], args.fraction, args.seed)
    model_config = TransformerConfig(
        vocab_size=len(vocabulary.tokens), block_size=args.block_size, d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout,
    )
    _, history, metadata = fit_transformer(
        subset, splits["validation"], vocabulary,
        args.artifacts_dir / f"transformer_{_scale_label(args.fraction)}.pt",
        epochs=args.epochs, batch_size=args.batch_size, seed=args.seed, device=args.device,
        learning_rate=args.learning_rate, config=model_config,
    )
    print(json.dumps({"metadata": metadata, "history": history}, indent=2))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    splits = _load_splits(args.data_dir)
    label = _scale_label(args.fraction)
    ngram_path = args.artifacts_dir / f"ngram_{label}.json"
    transformer_path = args.artifacts_dir / f"transformer_{label}.pt"
    if not ngram_path.exists() or not transformer_path.exists():
        raise FileNotFoundError("both trained models are required; run train-ngram and train-transformer first")
    ngram, transformer, vocabulary, _ = restore_models(ngram_path, transformer_path, str(choose_device(args.device)))
    ngram_eval = evaluate_predictor(ngram.predict_proba, splits["test"], vocabulary)
    model_device = next(transformer.parameters()).device
    transformer_eval = evaluate_predictor(
        lambda prefix: transformer.predict_proba(prefix, vocabulary.product_ids, model_device), splits["test"], vocabulary
    )
    evaluations = {"ngram": ngram_eval, "transformer": transformer_eval}
    payload = {
        "ngram": public_metrics(ngram_eval),
        "transformer": public_metrics(transformer_eval),
        "error_analysis": summarize_errors(evaluations, vocabulary),
    }
    write_json(args.results_dir / "metrics.json", payload)
    rows = []
    for metric in ("top_1_accuracy", "top_3_accuracy", "top_5_accuracy", "mrr", "mean_rank", "cross_entropy"):
        rows.append({"metric": metric, **{name: values[metric] for name, values in payload.items() if name in evaluations}})
    args.results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.results_dir / "model_comparison.csv", index=False)
    print(json.dumps(payload, indent=2))
    return 0


def _compare(args: argparse.Namespace) -> int:
    table = comparison_table(args.results_dir)
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    return 0


def _predict(args: argparse.Namespace) -> int:
    if args.top_k <= 0:
        raise ValueError("top-k must be positive")
    label = _scale_label(args.fraction)
    ngram_path = args.artifacts_dir / f"ngram_{label}.json"
    transformer_path = args.artifacts_dir / f"transformer_{label}.pt"
    if not ngram_path.exists() or not transformer_path.exists():
        raise FileNotFoundError("trained model files not found; run `python -m app.cli run-all` first")
    ngram, transformer, vocabulary, _ = restore_models(ngram_path, transformer_path, str(choose_device(args.device)))
    names = {product.name.casefold(): product.product_id for product in CATALOGUE}
    ids = set(vocabulary.tokens)
    items = []
    for raw in args.basket.split(","):
        value = raw.strip()
        product_id = value if value in ids else names.get(value.casefold())
        if product_id is None:
            raise ValueError(f"unknown product '{value}'; use a catalogue name or product ID")
        if product_id in (vocabulary.PAD, vocabulary.UNK):
            raise ValueError("special vocabulary tokens are not products")
        items.append(vocabulary.token_to_id[product_id])
    if not items:
        raise ValueError("basket must contain at least one product")
    transformer_device = next(transformer.parameters()).device
    rows = {}
    for model_name, predictor in (
        ("ngram", ngram.predict_proba),
        ("transformer", lambda prefix: transformer.predict_proba(prefix, vocabulary.product_ids, transformer_device)),
    ):
        top = predictor(items)
        ranked = sorted(top.items(), key=lambda pair: (-pair[1], pair[0]))[:args.top_k]
        rows[model_name] = [
            {"product_id": vocabulary.tokens[token_id], "probability": round(probability, 4)}
            for token_id, probability in ranked
        ]
    print(json.dumps({"basket": [vocabulary.tokens[item] for item in items], "predictions": rows}, indent=2))
    return 0


def _run_all(args: argparse.Namespace) -> int:
    vocabulary = Vocabulary.from_catalogue()
    model_config = TransformerConfig(
        vocab_size=len(vocabulary.tokens), block_size=args.block_size, d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout,
    )
    report = run_experiment(
        count=args.transactions, seed=args.seed, epochs=args.epochs, fractions=args.fractions,
        data_dir=args.data_dir, artifacts_dir=args.artifacts_dir, results_dir=args.results_dir,
        batch_size=args.batch_size, device=args.device, config=model_config,
    )
    paths = create_visualizations(args.results_dir, args.artifacts_dir)
    compact = {
        "split_sizes": report["split_sizes"],
        "training_fractions": list(report["training_sizes"]),
        "figures": [str(path) for path in paths],
    }
    print(json.dumps(compact, indent=2))
    return 0


def _visualize(args: argparse.Namespace) -> int:
    paths = create_visualizations(args.results_dir, args.artifacts_dir)
    print("\n".join(str(path) for path in paths))
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
import random
from pathlib import Path

from app.data import Session
from app.ngram import NGramModel
from app.transformer import TransformerConfig, load_checkpoint, save_checkpoint, train_transformer
from app.vocab import Vocabulary


def fit_ngram(
    sessions: list[Session], vocabulary: Vocabulary, path: Path, order: int = 3, alpha: float = 0.2
) -> tuple[NGramModel, dict[str, object]]:
    if not sessions:
        raise ValueError("training split is empty")
    sequences = [vocabulary.encode(session.items) for session in sessions]
    model = NGramModel(len(vocabulary.tokens), vocabulary.product_ids, order=order, alpha=alpha).fit(sequences)
    model.save(path)
    metadata = {"order": order, "alpha": alpha, "train_sessions": len(sessions), "vocabulary_size": len(vocabulary.tokens)}
    path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return model, metadata


def fit_transformer(
    train_sessions: list[Session],
    validation_sessions: list[Session],
    vocabulary: Vocabulary,
    path: Path,
    epochs: int = 8,
    batch_size: int = 64,
    learning_rate: float = 0.002,
    seed: int = 1729,
    device: str = "auto",
    config: TransformerConfig | None = None,
) -> tuple[object, dict[str, list[float]], dict[str, object]]:
    model, history, metadata = train_transformer(
        train_sessions, validation_sessions, vocabulary, config=config, epochs=epochs,
        batch_size=batch_size, learning_rate=learning_rate, seed=seed, device=device,
    )
    save_checkpoint(model, path, vocabulary, metadata)
    path.with_suffix(".history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    return model, history, metadata


def choose_training_subset(sessions: list[Session], fraction: float, seed: int) -> list[Session]:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if not sessions:
        raise ValueError("cannot select from an empty training split")
    count = max(1, round(len(sessions) * fraction))
    return random.Random(seed).sample(sessions, count)


def restore_models(ngram_path: Path, transformer_path: Path, device: str = "cpu"):
    ngram = NGramModel.load(ngram_path)
    transformer, vocabulary, metadata = load_checkpoint(transformer_path, device=device)
    if ngram.product_ids != vocabulary.product_ids:
        raise ValueError("saved models use different product vocabularies")
    return ngram, transformer, vocabulary, metadata

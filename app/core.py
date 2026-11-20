"""Small compatibility surface for the original milestone-based starter."""
from __future__ import annotations

from app.data import Session, generate_sessions, split_sessions
from app.experiment import run_experiment
from app.metrics import evaluate_predictor
from app.ngram import NGramModel
from app.transformer import RetailTransformer, TransformerConfig
from app.vocab import Vocabulary


def milestone_1(count: int = 1200, seed: int = 1729) -> list[Session]:
    """Create simulated shopping sessions with fixed random seeds."""
    return generate_sessions(count=count, seed=seed)


def milestone_2(sequences, vocabulary: Vocabulary, order: int = 3, alpha: float = 0.2) -> NGramModel:
    """Fit the statistical baseline to encoded training sessions."""
    return NGramModel(len(vocabulary.tokens), vocabulary.product_ids, order=order, alpha=alpha).fit(sequences)


def milestone_3(vocabulary: Vocabulary, config: TransformerConfig | None = None) -> RetailTransformer:
    """Initialize the small causal model used by the training pipeline."""
    import torch

    return RetailTransformer(config or TransformerConfig(vocab_size=len(vocabulary.tokens))).to(torch.device("cpu"))


def milestone_4(predictor, sessions: list[Session], vocabulary: Vocabulary):
    """Score predictions using the shared next-item metric implementation."""
    return evaluate_predictor(predictor, sessions, vocabulary)


def milestone_5(**options):
    """Run the shared split, training, evaluation, and reporting pipeline."""
    return run_experiment(**options)


def make_splits(sessions: list[Session], seed: int = 1729):
    """Split whole sessions before any next-item prefixes are expanded."""
    return split_sessions(sessions, seed=seed)

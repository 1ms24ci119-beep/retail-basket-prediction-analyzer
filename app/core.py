from app.data import DataError, Session, generate_sessions, split_sessions
from app.ngram import NGramModel
from app.vocab import Vocabulary


def milestone_1(count: int = 2400, seed: int = 1729):
    return generate_sessions(count=count, seed=seed)


def milestone_2(sequences, vocab: Vocabulary, order: int = 3, alpha: float = 0.2):
    return NGramModel(len(vocab.tokens), vocab.product_ids, order=order, alpha=alpha).fit(sequences)


def milestone_3():
    raise NotImplementedError("Use app.transformer for the neural language model")


def milestone_4():
    raise NotImplementedError("Use app.metrics for shared prediction metrics")


def milestone_5():
    raise NotImplementedError("Use app.experiment and app.visualize for comparison")

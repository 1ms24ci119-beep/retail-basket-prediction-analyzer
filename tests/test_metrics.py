import math

import pytest

from app.data import Session
from app.metrics import evaluate_predictor, prediction_rows, summarize_errors
from app.vocab import Vocabulary


def _confident_predictor(vocabulary, desired):
    def predict(prefix):
        probabilities = {item: 0.1 / (len(vocabulary.product_ids) - 1) for item in vocabulary.product_ids}
        probabilities[desired[len(prefix) - 1]] = 0.9
        return probabilities
    return predict


def test_prediction_prefixes_are_expanded_from_one_session():
    vocabulary = Vocabulary.from_catalogue()
    session = Session("one", ("bakery_bread", "dairy_butter", "snack_jam"), "breakfast")
    rows = list(prediction_rows([session], vocabulary))
    assert [len(row[0]) for row in rows] == [1, 2]
    assert {row[2].session_id for row in rows} == {"one"}


def test_top_k_mrr_rank_and_cross_entropy_are_calculated():
    vocabulary = Vocabulary.from_catalogue()
    session = Session("one", ("bakery_bread", "dairy_butter", "snack_jam"), "breakfast")
    ids = vocabulary.encode(session.items)
    result = evaluate_predictor(_confident_predictor(vocabulary, ids[1:]), [session], vocabulary)
    assert result["examples"] == 2
    assert result["top_1_accuracy"] == 1.0
    assert result["top_3_accuracy"] == 1.0
    assert result["top_5_accuracy"] == 1.0
    assert result["mrr"] == 1.0
    assert result["mean_rank"] == 1.0
    assert result["cross_entropy"] == pytest.approx(-math.log(0.9))
    assert "dairy" in result["per_category_accuracy"]
    assert "snacks" in result["per_category_accuracy"]
    assert result["accuracy_by_prefix_size"]["1-2"] == 1.0


def test_evaluation_rejects_empty_and_malformed_distributions():
    vocabulary = Vocabulary.from_catalogue()
    session = Session("one", ("bakery_bread", "dairy_butter"), "breakfast")
    with pytest.raises(ValueError, match="empty"):
        evaluate_predictor(lambda prefix: {}, [], vocabulary)
    with pytest.raises(ValueError, match="full product vocabulary"):
        evaluate_predictor(lambda prefix: {2: 1.0}, [session], vocabulary)
    with pytest.raises(ValueError, match="sum to one"):
        evaluate_predictor(lambda prefix: {item: 0.5 for item in vocabulary.product_ids}, [session], vocabulary)


def test_error_analysis_aligns_models_and_reports_comparisons():
    vocabulary = Vocabulary.from_catalogue()
    session = Session("one", ("bakery_bread", "dairy_butter", "snack_jam"), "breakfast")
    ids = vocabulary.encode(session.items)
    good = evaluate_predictor(_confident_predictor(vocabulary, ids[1:]), [session], vocabulary)

    def wrong(prefix):
        probs = {item: 0.1 / (len(vocabulary.product_ids) - 1) for item in vocabulary.product_ids}
        probs[2] = 0.9
        return probs

    bad = evaluate_predictor(wrong, [session], vocabulary)
    report = summarize_errors({"ngram": good, "transformer": bad}, vocabulary)
    assert report["transformer_better_examples"] == 0
    assert report["ngram_better_examples"] >= 1
    assert report["different_rank_examples"] > 0

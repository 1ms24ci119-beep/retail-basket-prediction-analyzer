import json

import pytest

from app.data import (
    CATALOGUE,
    DataError,
    Session,
    describe_sessions,
    generate_sessions,
    load_sessions,
    save_sessions,
    split_sessions,
)
from app.ngram import NGramModel
from app.vocab import Vocabulary


def test_milestone_1_generation_is_seeded_and_valid():
    first = generate_sessions(60, seed=21)
    second = generate_sessions(60, seed=21)
    assert first == second
    assert all(2 <= len(session.items) <= 8 for session in first)
    assert all(set(session.items).issubset({product.product_id for product in CATALOGUE}) for session in first)


def test_generator_has_planted_purchase_order_and_mixed_patterns():
    sessions = generate_sessions(600, seed=29)
    assert {"toast_breakfast", "pasta_night", "independent", "unusual"}.issubset({row.pattern for row in sessions})
    assert any(
        row.pattern == "toast_breakfast"
        and list(row.items[:3]) == ["bakery_bread", "dairy_butter", "snack_jam"]
        for row in sessions
    )


def test_split_is_disjoint_repeatable_and_proportional():
    sessions = generate_sessions(100, seed=4)
    first = split_sessions(sessions, seed=4)
    second = split_sessions(sessions, seed=4)
    id_sets = {name: {session.session_id for session in group} for name, group in first.items()}
    assert first == second
    assert id_sets["train"].isdisjoint(id_sets["validation"])
    assert id_sets["train"].isdisjoint(id_sets["test"])
    assert id_sets["validation"].isdisjoint(id_sets["test"])
    assert sum(map(len, first.values())) == len(sessions)


def test_duplicate_session_ids_are_rejected():
    session = Session("same", ("bakery_bread", "dairy_butter"), "example")
    with pytest.raises(DataError, match="unique"):
        split_sessions([session] * 10)


def test_data_splits_round_trip_as_jsonl(tmp_path):
    expected = split_sessions(generate_sessions(60, seed=6), seed=6)
    save_sessions(expected, tmp_path)
    actual = {name: load_sessions(tmp_path / f"{name}.jsonl") for name in expected}
    assert actual == expected
    assert (tmp_path / "catalogue.json").exists()
    assert len(json.loads((tmp_path / "catalogue.json").read_text())) >= 40


def test_dataset_stats_report_products_lengths_and_pairs():
    sessions = generate_sessions(40, seed=1)
    summary = describe_sessions(sessions)
    assert summary["transactions"] == 40
    assert 40 <= summary["unique_products"] <= 60
    assert 2 <= summary["average_basket_length"] <= 8
    assert summary["most_common_pairs"]
    assert summary["most_common_itemsets"]


def test_small_invalid_datasets_are_rejected():
    with pytest.raises(DataError):
        generate_sessions(9)
    with pytest.raises(DataError):
        split_sessions(generate_sessions(10), train_fraction=0.8, validation_fraction=0.2)


def test_vocabulary_has_fixed_special_and_catalogue_ids():
    vocabulary = Vocabulary.from_catalogue()
    assert vocabulary.tokens[:2] == ("<PAD>", "<UNK>")
    assert len(set(vocabulary.tokens)) == len(vocabulary.tokens)
    assert vocabulary.token_to_id["dairy_milk"] == 2
    assert len(vocabulary.product_ids) >= 40


def test_vocabulary_encodes_decodes_unknown_and_padding():
    vocabulary = Vocabulary.from_catalogue()
    encoded = vocabulary.encode(["bakery_bread", "not_in_catalogue"])
    assert encoded == [vocabulary.token_to_id["bakery_bread"], 1]
    assert vocabulary.decode(encoded) == ["bakery_bread"]
    assert vocabulary.decode(encoded, include_special=True) == ["bakery_bread", "<UNK>"]
    assert vocabulary.pad([2, 3], 4) == [2, 3, 0, 0]
    assert vocabulary.pad([2, 3, 4], 2) == [3, 4]
    with pytest.raises(ValueError):
        vocabulary.pad([2, 3], 1, truncate=False)


def test_vocabulary_serialization_preserves_mapping():
    vocabulary = Vocabulary.from_catalogue()
    restored = Vocabulary.from_dict(vocabulary.to_dict())
    assert restored == vocabulary


def test_milestone_2_counts_unigrams_and_contexts():
    model = NGramModel(vocab_size=6, product_ids=[2, 3, 4, 5], order=2, alpha=0.1)
    model.fit([[2, 3, 4], [2, 3, 5]])
    assert model.counts[1][()][3] == 2
    assert model.counts[2][(2,)][3] == 2
    assert model.counts[2][(3,)][4] == 1


def test_ngram_distribution_is_normalized_and_top_item_follows_context():
    model = NGramModel(6, [2, 3, 4, 5], order=2, alpha=0.1).fit([[2, 3, 4]] * 20 + [[2, 3, 5]])
    probabilities = model.predict_proba([2, 3])
    assert set(probabilities) == {2, 3, 4, 5}
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert model.predict([2, 3], top_k=2)[0][0] == 4


def test_ngram_smoothing_handles_unseen_contexts_and_empty_prefix():
    model = NGramModel(5, [2, 3, 4], order=3, alpha=0.5).fit([[2, 3, 4]])
    start_probabilities = model.predict_proba([])
    assert sum(start_probabilities.values()) == pytest.approx(1.0)
    probabilities = model.predict_proba([4, 2])
    assert all(probability > 0 for probability in probabilities.values())
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_ngram_persistence_round_trip(tmp_path):
    model = NGramModel(5, [2, 3, 4], order=3).fit([[2, 3, 4], [2, 4, 3]])
    path = tmp_path / "model.json"
    model.save(path)
    restored = NGramModel.load(path)
    assert restored.predict_proba([2]) == pytest.approx(model.predict_proba([2]))


def test_ngram_rejects_invalid_configuration_and_input():
    with pytest.raises(ValueError):
        NGramModel(4, [2, 3], order=5)
    with pytest.raises(ValueError):
        NGramModel(4, [2, 3], alpha=0)
    model = NGramModel(4, [2, 3]).fit([[2, 3]])
    with pytest.raises(ValueError):
        model.predict_proba([1])
    with pytest.raises(ValueError):
        model.predict([2], top_k=0)

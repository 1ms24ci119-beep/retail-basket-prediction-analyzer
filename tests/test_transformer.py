import math

import torch
from torch.nn import functional as F

from app.data import generate_sessions, split_sessions
from app.transformer import (
    CausalSelfAttention,
    RetailTransformer,
    TransformerConfig,
    load_checkpoint,
    save_checkpoint,
    train_transformer,
)
from app.vocab import Vocabulary


def small_config(vocab_size):
    return TransformerConfig(vocab_size=vocab_size, block_size=8, d_model=16, n_heads=2, n_layers=2, dropout=0.0)


def test_transformer_initializes_with_compact_parameter_count():
    vocabulary = Vocabulary.from_catalogue()
    model = RetailTransformer(small_config(len(vocabulary.tokens)))
    assert sum(parameter.numel() for parameter in model.parameters()) < 100_000


def test_forward_shape_and_causal_mask_prevents_future_token_leakage():
    torch.manual_seed(3)
    vocabulary = Vocabulary.from_catalogue()
    model = RetailTransformer(small_config(len(vocabulary.tokens))).eval()
    first = torch.tensor([[2, 3, 4]])
    changed_future = torch.tensor([[2, 3, 7]])
    logits_first = model(first)
    logits_second = model(changed_future)
    assert logits_first.shape == (1, 3, len(vocabulary.tokens))
    assert torch.allclose(logits_first[:, :2], logits_second[:, :2], atol=1e-6)
    assert torch.equal(model.blocks[0].attention.causal_mask[:4, :4], torch.tril(torch.ones(4, 4, dtype=torch.bool)))


def test_attention_respects_padding_keys():
    torch.manual_seed(4)
    config = small_config(12)
    attention = CausalSelfAttention(config).eval()
    values = torch.randn(1, 3, config.d_model)
    left_padded = values.clone()
    left_padded[:, 0] = 900
    valid = torch.tensor([[False, True, True]])
    result = attention(left_padded, valid)
    changed = left_padded.clone()
    changed[:, 0] = -900
    changed_result = attention(changed, valid)
    assert torch.allclose(result[:, 1:], changed_result[:, 1:], atol=1e-6)


def test_loss_is_finite_for_shifted_product_targets():
    vocabulary = Vocabulary.from_catalogue()
    model = RetailTransformer(small_config(len(vocabulary.tokens)))
    sequence = torch.tensor([[2, 3, 4, 0]])
    logits = model(sequence[:, :-1])
    logits[..., :2] = torch.finfo(logits.dtype).min
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), sequence[:, 1:].reshape(-1), ignore_index=0)
    assert math.isfinite(float(loss.detach()))


def test_checkpoint_saves_and_loads_predictions(tmp_path):
    vocabulary = Vocabulary.from_catalogue()
    model = RetailTransformer(small_config(len(vocabulary.tokens)))
    path = tmp_path / "model.pt"
    save_checkpoint(model, path, vocabulary, {"seed": 5})
    restored, restored_vocabulary, metadata = load_checkpoint(path)
    assert restored_vocabulary == vocabulary
    assert metadata["seed"] == 5
    assert restored.predict([2, 3], vocabulary.product_ids, top_k=4)


def test_one_epoch_training_reduces_or_tracks_losses_and_is_predictable():
    vocabulary = Vocabulary.from_catalogue()
    splits = split_sessions(generate_sessions(60, seed=2), seed=2)
    model, history, metadata = train_transformer(
        splits["train"], splits["validation"], vocabulary,
        config=small_config(len(vocabulary.tokens)), epochs=1, batch_size=16, seed=11, device="cpu",
    )
    assert len(history["train_loss"]) == len(history["validation_loss"]) == 1
    assert all(math.isfinite(value) for values in history.values() for value in values)
    assert metadata["parameters"] > 0
    assert abs(sum(model.predict_proba([2], vocabulary.product_ids).values()) - 1) < 1e-6

import numpy as np
import pytest

import tensorforge as tf
from tensorforge.evaluation import classification_accuracy, evaluate_language_model
from tensorforge.models import CausalLanguageModel, LanguageModelConfig
from tensorforge.pretrained import from_pretrained, save_pretrained
from tensorforge.text import BytePairTokenizer, CharacterTokenizer, instruction_sample, mask_tokens


def test_bpe_roundtrip_merges_and_state():
    text = "hello hello hello café 👋\n" * 4
    tokenizer = BytePairTokenizer(text, 290)
    assert tokenizer.vocab_size > 260
    assert len(tokenizer.encode(text)) < len(text.encode("utf-8"))
    for value in (text, "new Ω characters 🌍", "", "\x00"):
        assert tokenizer.decode(tokenizer.encode(value)) == value
    clone = BytePairTokenizer.from_state_dict(tokenizer.state_dict())
    assert clone.encode(text) == tokenizer.encode(text)
    with pytest.raises(ValueError):
        BytePairTokenizer.from_state_dict({"type": "byte_pair", "merges": [[4, 999]]})


def test_masking_specials_and_instruction_targets():
    tokens = tf.tensor([[0, 4, 5, 6, 7], [0, 8, 9, 4, 5]], dtype=np.int64)
    corrupted, targets = mask_tokens(
        tokens, mask_id=3, vocab_size=10, probability=1, special_ids=(0, 3), seed=4
    )
    np.testing.assert_equal(targets.numpy()[:, 0], -100)
    np.testing.assert_equal(targets.numpy()[:, 1:], tokens.numpy()[:, 1:])
    assert corrupted.dtype == np.int64
    x, y = mask_tokens(tokens, 3, 10, probability=0, seed=4)
    np.testing.assert_equal(x.numpy(), tokens.numpy())
    np.testing.assert_equal(y.numpy(), -100)
    tokenizer = CharacterTokenizer("QuestionAnswer")
    inputs, labels = instruction_sample(tokenizer, "Question", "Answer")
    assert inputs.shape == labels.shape
    np.testing.assert_equal(labels.numpy()[: len("Question")], -100)
    assert labels.numpy()[-1] == tokenizer.eos_id


def test_bundle_and_evaluation(tmp_path):
    tokenizer = CharacterTokenizer("abc")
    model = CausalLanguageModel(
        LanguageModelConfig(
            tokenizer.vocab_size, features=4, num_heads=1, num_layers=1, hidden_size=8, max_length=8
        )
    )
    path = tmp_path / "model.npz"
    save_pretrained(model, path, tokenizer)
    clone, loaded_tokenizer = from_pretrained(path)
    x = tf.tensor([[4, 5, 6]], dtype=np.int64)
    np.testing.assert_equal(model(x).numpy(), clone(x).numpy())
    assert loaded_tokenizer.encode("abc") == tokenizer.encode("abc")
    assert not clone.training
    model.train()
    metrics = evaluate_language_model(model, [(x, x), (x[:, :1], x[:, :1])])
    assert metrics["tokens"] == 4 and metrics["perplexity"] == pytest.approx(
        np.exp(metrics["loss"])
    )
    assert model.training
    assert classification_accuracy(tf.tensor([[1, 0], [0, 1]]), np.array([0, 1])) == 1
    with pytest.raises(ValueError):
        evaluate_language_model(model, [(x, tf.tensor([[-100] * 3], dtype=np.int64))])


def test_merged_lora_bundle_roundtrip(tmp_path):
    from tensorforge.finetuning import apply_lora, merge_lora, trainable_parameters

    model = CausalLanguageModel(
        LanguageModelConfig(8, features=4, num_heads=1, num_layers=1, hidden_size=8, max_length=8)
    )
    apply_lora(model, rank=2, alpha=2)
    for parameter in trainable_parameters(model):
        parameter._assign(np.full(parameter.shape, 0.1))
    tokens = tf.tensor([[4, 5, 6]], dtype=np.int64)
    expected = model(tokens).numpy()
    path = tmp_path / "adapted.npz"
    with pytest.raises(ValueError, match="Merge LoRA"):
        save_pretrained(model, path)
    merge_lora(model)
    save_pretrained(model, path)
    restored, tokenizer = from_pretrained(path)
    assert tokenizer is None
    np.testing.assert_allclose(restored(tokens).numpy(), expected, atol=1e-12)

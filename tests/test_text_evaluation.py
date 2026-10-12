import numpy as np
import pytest

import tensorforge as tf
from tensorforge.evaluation import classification_accuracy, evaluate_language_model
from tensorforge.models import CausalLanguageModel, LanguageModelConfig
from tensorforge.pretrained import from_pretrained, save_pretrained
from tensorforge.text import BytePairTokenizer, CharacterTokenizer


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


@pytest.mark.parametrize("tokenizer_class", [CharacterTokenizer, BytePairTokenizer])
def test_bundle_and_evaluation(tmp_path, tokenizer_class):
    tokenizer = tokenizer_class("abc")
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

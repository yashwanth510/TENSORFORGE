import numpy as np

import tensorforge as tf
from tensorforge import nn
from tensorforge.finetuning import apply_lora, trainable_parameters
from tensorforge.generation import beam_search, speculative_greedy
from tensorforge.models import CausalLanguageModel, LanguageModelConfig
from tensorforge.retrieval import DocumentIndex
from tensorforge.text import CharacterTokenizer, PairedTextDataset, corrupt_spans


def test_masked_language_model_and_padding():
    tf.manual_seed(7)
    model = nn.MaskedLanguageModel(12, features=8, num_heads=2, num_layers=1)
    ids = tf.tensor([[4, 3, 5], [6, 3, 7]], dtype=np.int64)
    logits = model(ids)
    assert logits.shape == (2, 3, 12)
    target = tf.tensor([[-100, 4, -100], [-100, 6, -100]], dtype=np.int64)
    nn.CrossEntropyLoss()(logits, target).backward()
    assert all(p.grad is not None and np.isfinite(p.grad).all() for p in model.parameters())


def test_vision_and_multimodal_gradients(gradcheck):
    tf.manual_seed(5)
    patches = nn.PatchEmbedding(1, 2, 4)
    images = np.arange(16.0).reshape(1, 1, 4, 4) / 16
    gradcheck(patches, [images])
    vision = nn.VisionTransformer(4, 2, 1, 2, features=4, num_heads=1)
    x = tf.tensor(images, requires_grad=True)
    vision(x).sum().backward()
    assert np.isfinite(x.grad).all()
    model = nn.MultimodalClassifier(10, 1, 2, 2, features=4, num_heads=1)
    output = model(
        tf.tensor(images), np.array([[1, 2, 3]]), text_padding_mask=np.array([[False, False, True]])
    )
    output.sum().backward()
    assert output.shape == (1, 2)
    assert all(p.grad is not None for p in model.parameters())


def test_expert_routing_and_gradients(gradcheck):
    tf.manual_seed(15)
    model = nn.MixtureOfExperts(2, 4, num_experts=3, top_k=2)
    x = np.array([[0.3, -0.7], [1.1, 0.2]])
    gradcheck(model, [x])
    model.zero_grad()
    output, aux = model(tf.tensor(x), return_aux_loss=True)
    ((output * output).mean() + 0.01 * aux).backward()
    assert model.router.weight.grad is not None
    assert aux.item() > 0


def test_lora_in_language_model():
    model = CausalLanguageModel(
        LanguageModelConfig(8, features=4, num_heads=1, num_layers=1, hidden_size=8, max_length=8)
    )
    names = apply_lora(model, rank=2, alpha=2)
    assert len(names) == 2 and len(trainable_parameters(model)) == 4
    x = tf.tensor([[1, 2, 3]], dtype=np.int64)
    nn.CrossEntropyLoss()(model(x), x).backward()
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert all(p.grad is not None for p in trainable_parameters(model))


def test_beam_constraints_and_speculation():
    tf.manual_seed(2)
    config = LanguageModelConfig(
        7, features=4, num_heads=1, num_layers=1, hidden_size=8, max_length=16
    )
    target, draft = CausalLanguageModel(config), CausalLanguageModel(config)
    prompt = np.array([[1, 2]])
    expected = target.generate(prompt, max_new_tokens=5, temperature=0, use_cache=False)
    np.testing.assert_equal(beam_search(target, prompt, 5, num_beams=1).numpy(), expected.numpy())
    for steps in (1, 2, 4):
        np.testing.assert_equal(
            speculative_greedy(target, draft, prompt, 5, draft_steps=steps).numpy(),
            expected.numpy(),
        )
    restricted = beam_search(target, prompt, 3, allowed_tokens_fn=lambda prefix: [4])
    np.testing.assert_equal(restricted.numpy()[0, 2:], [4, 4, 4])
    assert target.training and draft.training


def test_retrieval_and_paired_data():
    index = DocumentIndex(
        {"a": "Cats like naps.", "b": "Transformers use attention.", "c": "Dogs bark."}
    )
    assert index.search("attention")[0]["id"] == "b"
    assert index.search("unrelated") == []
    prompt, hits = index.prompt("attention")
    assert "[b]" in prompt and len(hits) == 1
    tokenizer = CharacterTokenizer("hello bonjour")
    dataset = PairedTextDataset([("hello", "bonjour")], tokenizer)
    source, x, y = dataset[0]
    assert source.numpy()[-1] == tokenizer.eos_id
    np.testing.assert_equal(x.numpy()[1:], y.numpy()[:-1])
    source, target = corrupt_spans(np.arange(10), range(20, 30), noise_density=0.5, seed=2)
    reconstruction = {}
    active = None
    for token in target.numpy():
        if token >= 20:
            active = int(token)
            reconstruction[active] = []
        else:
            reconstruction[active].append(int(token))
    restored = []
    for token in source.numpy():
        restored.extend(reconstruction[int(token)] if token >= 20 else [int(token)])
    assert restored == list(range(10))

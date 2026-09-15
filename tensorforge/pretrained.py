"""Self-contained TensorForge model bundles; no third-party weight conversion."""

from .models import CausalLanguageModel, LanguageModelConfig
from .serialization import load, save
from .text import BytePairTokenizer, CharacterTokenizer


def save_pretrained(model, path, tokenizer=None):
    from .finetuning import LoRALinear

    if not isinstance(model, CausalLanguageModel):
        raise TypeError("Model bundles currently support CausalLanguageModel.")
    if any(isinstance(layer, LoRALinear) for _, layer in model._walk()):
        raise ValueError("Merge LoRA layers before saving a native model bundle.")
    if tokenizer is not None and not isinstance(tokenizer, (CharacterTokenizer, BytePairTokenizer)):
        raise TypeError("Unsupported tokenizer type.")
    if tokenizer is not None and tokenizer.vocab_size != model.config.vocab_size:
        raise ValueError("Tokenizer and model vocabulary sizes must match.")
    save(
        {
            "format": "tensorforge-model",
            "version": 1,
            "config": model.config.to_dict(),
            "weights": model.state_dict(),
            "tokenizer": None if tokenizer is None else tokenizer.state_dict(),
        },
        path,
    )


def from_pretrained(path):
    """Return (model, tokenizer_or_none); model starts in evaluation mode."""
    state = load(path)
    if state.get("format") != "tensorforge-model" or state.get("version") != 1:
        raise ValueError("Unsupported model bundle.")
    model = CausalLanguageModel(LanguageModelConfig(**state["config"]))
    model.load_state_dict(state["weights"])
    model.eval()
    tokenizer = None
    if state["tokenizer"] is not None:
        kind = state["tokenizer"].get("type")
        tokenizers = {"character": CharacterTokenizer, "byte_pair": BytePairTokenizer}
        if kind not in tokenizers:
            raise ValueError("Unsupported tokenizer format.")
        tokenizer = tokenizers[kind].from_state_dict(state["tokenizer"])
        if tokenizer.vocab_size != model.config.vocab_size:
            raise ValueError("Tokenizer and model vocabulary sizes differ.")
    return model, tokenizer

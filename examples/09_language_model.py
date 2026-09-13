"""Train GPT or a modern decoder on local text, save it, and generate text.

python examples/09_language_model.py --variant modern --checkpoint checkpoints/modern.npz
python examples/09_language_model.py --load checkpoints/modern.npz --prompt 'hello '
"""

import argparse
from pathlib import Path

import tensorforge as tf
from tensorforge import nn, optim
from tensorforge.data import DataLoader
from tensorforge.models import CausalLanguageModel, LanguageModelConfig
from tensorforge.text import CharacterTokenizer, TextDataset
from tensorforge.training import LRScheduler, clip_grad_norm_, save_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("gpt", "modern"), default="gpt")
    parser.add_argument(
        "--text-file", type=Path, help="UTF-8 training text; defaults to a tiny built-in sample"
    )
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument(
        "--checkpoint", type=Path, help="Where to save the trained model and training state"
    )
    parser.add_argument(
        "--load", type=Path, help="Load an existing checkpoint for generation without training"
    )
    parser.add_argument("--prompt", default="hello ")
    parser.add_argument("--new-tokens", type=int, default=24)
    args = parser.parse_args()
    tf.manual_seed(21)

    if args.load:
        state = tf.load(args.load)
        extra = state["extra"]
        tokenizer = CharacterTokenizer.from_state_dict(extra["tokenizer"])
        model = CausalLanguageModel(LanguageModelConfig(**extra["config"]))
        model.load_state_dict(state["model"])
    else:
        if args.steps <= 0:
            parser.error("--steps must be positive")
        text = (
            args.text_file.read_text(encoding="utf-8")
            if args.text_file
            else "hello tensorforge.\n" * 40
        )
        if len(text) < 3:
            parser.error("training text must contain at least three characters")
        tokenizer = CharacterTokenizer(text)
        sequence_length = min(32, len(text) - 1)
        dataset = TextDataset(tokenizer.encode(text, add_eos=True), sequence_length)
        loader = DataLoader(dataset, batch_size=8, shuffle=True, seed=21)
        config = LanguageModelConfig(
            tokenizer.vocab_size,
            features=16,
            num_heads=2,
            num_layers=1,
            hidden_size=32,
            max_length=64,
            variant=args.variant,
            num_kv_heads=1 if args.variant == "modern" else None,
        )
        model = CausalLanguageModel(config)
        optimizer = optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.01)
        scheduler = LRScheduler(
            optimizer,
            schedule="cosine",
            warmup_steps=min(5, args.steps - 1),
            total_steps=args.steps,
            min_lr=0.002,
        )
        fixed_x, fixed_y = next(iter(DataLoader(dataset, batch_size=8)))
        initial = nn.CrossEntropyLoss()(model(fixed_x), fixed_y).item()
        batches = iter(loader)
        for step in range(args.steps):
            try:
                x, y = next(batches)
            except StopIteration:
                batches = iter(loader)
                x, y = next(batches)
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        model.eval()
        with tf.no_grad():
            final = nn.CrossEntropyLoss()(model(fixed_x), fixed_y).item()
        print(f"{args.variant}: next-token loss {initial:.6f} -> {final:.6f}")
        if not args.text_file and args.steps >= 120:
            assert final < initial * 0.5
        if args.checkpoint:
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                args.checkpoint,
                model,
                optimizer,
                scheduler,
                loader,
                step=args.steps,
                extra={"config": config.to_dict(), "tokenizer": tokenizer.state_dict()},
            )
            print("Checkpoint:", args.checkpoint)

    prompt = tokenizer.encode(args.prompt)
    if not prompt:
        prompt = [tokenizer.bos_id]
    output = model.generate(
        [prompt], max_new_tokens=args.new_tokens, temperature=0, eos_token_id=tokenizer.eos_id
    )
    print("Generated:", repr(tokenizer.decode(output.numpy()[0])))


if __name__ == "__main__":
    main()

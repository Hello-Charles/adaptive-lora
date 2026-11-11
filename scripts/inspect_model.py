"""Inspect Qwen layer structure and one short forward pass without ablation."""

import argparse
from importlib import metadata
import json
from pathlib import Path
import subprocess

from adaptive_lora.model_loader import (
    DEFAULT_MODEL_ID,
    EXPECTED_LAYERS,
    describe_output,
    discover_layer_modules,
    load_model_and_tokenizer,
    prediction_position,
)


def inspect(model, tokenizer, text):
    import torch

    layers = discover_layer_modules(model)
    inputs = tokenizer(text, return_tensors="pt", padding=False, truncation=False, add_special_tokens=True)
    position = prediction_position(inputs["input_ids"], inputs.get("attention_mask"))
    observed = {}
    handles = []

    def capture(index, component):
        def record(_module, _inputs, output):
            observed[(index, component)] = describe_output(output)
        return record

    try:
        for layer in layers:
            handles.append(layer.attention_module.register_forward_hook(capture(layer.index, "attention")))
            handles.append(layer.mlp_module.register_forward_hook(capture(layer.index, "mlp")))
        model.eval()
        with torch.inference_mode():
            result = model(**inputs)
    finally:
        for handle in handles:
            handle.remove()

    logits = result.logits
    if len(logits.shape) != 3 or logits.shape[0] != 1 or logits.shape[1] != inputs["input_ids"].shape[1]:
        raise ValueError(f"Unexpected logits shape: {list(logits.shape)}")
    if len(observed) != EXPECTED_LAYERS * 2:
        raise ValueError(f"Expected outputs from both submodules in {EXPECTED_LAYERS} layers; found {len(observed)}")

    return {
        "model_type": type(model).__name__,
        "configured_layers": model.config.num_hidden_layers,
        "resolved_revision": getattr(model.config, "_commit_hash", None),
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "is_fast": getattr(tokenizer, "is_fast", None),
            "padding_side": tokenizer.padding_side,
            "truncation_side": tokenizer.truncation_side,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        },
        "short_input": {
            "text": text,
            "input_ids": inputs["input_ids"][0].tolist(),
            "prediction_position": position,
            "logits_shape": list(logits.shape),
            "predicted_token_id": int(logits[0, position].argmax().item()),
        },
        "layers": [
            {
                "index": layer.index,
                "block": layer.block_path,
                "attention": layer.attention_path,
                "mlp": layer.mlp_path,
                "projections": layer.projections,
                "attention_output": observed[(layer.index, "attention")],
                "mlp_output": observed[(layer.index, "mlp")],
            }
            for layer in layers
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", help="Pinned model revision, if known")
    parser.add_argument("--text", default="Hello, world!")
    parser.add_argument("--output", type=Path, required=True, help="New report path (never overwritten)")
    parser.add_argument("--allow-download", action="store_true", help="Permit downloading model weights when explicitly authorized")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")

    try:
        model, tokenizer = load_model_and_tokenizer(
            args.model_id, revision=args.revision, allow_download=args.allow_download
        )
    except OSError as exc:
        if not args.allow_download:
            parser.exit(2, f"Model unavailable locally; no weights downloaded: {exc}\n")
        raise

    report = inspect(model, tokenizer, args.text)
    report.update(
        model_id=args.model_id,
        requested_revision=args.revision,
        packages={name: metadata.version(name) for name in ("torch", "transformers", "peft")},
        git_commit=subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip() or None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, ensure_ascii=False)
        output.write("\n")
    print(f"Structure report written: {args.output}")


if __name__ == "__main__":
    main()

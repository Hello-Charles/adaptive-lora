"""单样本模型结构探查；观察 hook 不得替换或原地改动子模块输出。

命令默认离线。下载需显式授权标志、revision 和新的独立缓存目录；报告拒绝覆盖。
脚本应从项目目录运行，不应把下载目录作为工作目录或 Python 脚本搜索路径。
"""

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
    """对兼容 Qwen 结构的模型记录一次单样本前向的形状和路径。

    调用者应传入短文本及彼此兼容的模型/tokenizer；此处不截断、不 padding，
    也不自动迁移输入设备，因而默认适用于 CPU 加载路径。模型会保持 eval 状态。
    子模块只保存输出元信息，不保留激活张量，不执行干预；异常时也必须移除 hook。
    """
    import torch

    layers = discover_layer_modules(model)
    inputs = tokenizer(text, return_tensors="pt", padding=False, truncation=False, add_special_tokens=True)
    position = prediction_position(inputs["input_ids"], inputs.get("attention_mask"))
    observed = {}
    handles = []

    def capture(index, component):
        """每个回调绑定自己的层号/组件名，避免循环闭包都引用最后一层。"""
        def record(_module, _inputs, output):
            # 观察 hook 必须返回 None；返回非 None 会替换模块输出，改变被测前向。
            observed[(index, component)] = describe_output(output)
        return record

    # 注册阶段也可能在中途失败；已成功注册的观察者不能泄漏到模型后续前向中。
    try:
        for layer in layers:
            handles.append(layer.attention_module.register_forward_hook(capture(layer.index, "attention")))
            handles.append(layer.mlp_module.register_forward_hook(capture(layer.index, "mlp")))
        model.eval()
        # eval 控制 dropout 等模块行为；inference_mode 才禁止本次前向记录自动求导图。
        with torch.inference_mode():
            result = model(**inputs)
    finally:
        for handle in handles:
            handle.remove()

    # logits 的序列维须保留全部输入位置，不能是生成接口只返回的最后一列。
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
    """下载开关只表示许可；可重复实验仍须由调用者提供不可变提交 SHA。

    下载只能用新缓存目录；已有缓存的复核应走默认离线路径。
    输出在探查成功后独占创建，模型加载或前向失败不应留下成功报告。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", help="Pinned model revision, if known")
    parser.add_argument("--cache-dir", type=Path, help="Dedicated model cache directory")
    parser.add_argument("--text", default="Hello, world!")
    parser.add_argument("--output", type=Path, required=True, help="New report path (never overwritten)")
    parser.add_argument("--allow-download", action="store_true", help="Permit downloading model weights when explicitly authorized")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    if args.allow_download:
        if args.revision is None or args.cache_dir is None:
            parser.error("Downloading requires both --revision and --cache-dir")
        try:
            args.cache_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            parser.error(f"Download cache must be a new, empty directory: {args.cache_dir}")

    try:
        model, tokenizer = load_model_and_tokenizer(
            args.model_id, revision=args.revision, allow_download=args.allow_download, cache_dir=args.cache_dir
        )
    except OSError as exc:
        if not args.allow_download:
            parser.exit(2, f"Model unavailable locally; no weights downloaded: {exc}\n")
        raise

    report = inspect(model, tokenizer, args.text)
    report.update(
        model_id=args.model_id,
        requested_revision=args.revision,
        cache_dir=str(args.cache_dir) if args.cache_dir is not None else None,
        packages={name: metadata.version(name) for name in ("torch", "transformers", "peft")},
        # HEAD 只是版本基线，不包含工作区改动；报告中的 SHA 不能证明代码树是干净的。
        git_commit=subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip() or None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # 预检查与写文件之间也可能出现同名文件，独占创建确保历史报告不会被覆盖。
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, ensure_ascii=False)
        output.write("\n")
    print(f"Structure report written: {args.output}")


if __name__ == "__main__":
    main()

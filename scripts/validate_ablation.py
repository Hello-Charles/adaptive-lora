"""M3 的小输入离线验证：独立消融、输出结构、恢复、异常清理及基础参数字节指纹。

只访问固定本地模型缓存；不下载、不训练、不算预测 token/概率/层重要性。
默认只验证第 0 层；--all-layers 才遍历 24 层的 Attention/MLP，始终一次一个目标。
"""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import re
import subprocess
import sys
from time import perf_counter

import torch

from adaptive_lora.ablation import zero_ablation
from adaptive_lora.model_loader import (
    DEFAULT_MODEL_ID, EXPECTED_LAYERS, describe_output, discover_layer_modules,
    load_model_and_tokenizer, prediction_position,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
MODEL_CACHE = ROOT / "data/models/qwen2_5_0_5b_060db6499f32faf8b98477b0a26969ef7d8b9987"


class InjectedForwardFailure(RuntimeError):
    """在真实目标输出检查完成后中断前向，只用于验证上下文的 finally 路径。"""


def check(condition, message):
    """验收失败立即抛异常；不依赖可被 python -O 关闭的 assert 语句。"""
    if not condition:
        raise AssertionError(message)


def parameter_fingerprint(model):
    """对 CPU 参数名、dtype、shape 和原始字节求 SHA-256；不包含 buffers 或 hook 状态。

    uint8 view 可表示 bfloat16 原始字节，memoryview 避免把巨大 embedding 再复制成 bytes。
    非连续参数可能临时复制为连续布局；本工具限小型 CPU smoke 模型，不用于训练热路径。
    """
    digest = hashlib.sha256()
    count = 0
    for name, value in model.named_parameters():
        check(value.device.type == "cpu", "This validator only supports CPU parameters")
        header = json.dumps([name, str(value.dtype), list(value.shape)], ensure_ascii=False).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        raw = value.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
        digest.update(memoryview(raw).cast("B"))
        count += value.numel()
    return {"sha256": digest.hexdigest(), "parameter_elements": count}


def hook_inventory(model):
    """仅用于验证本地 forward hook 生命周期，保留 ID 和回调身份而非只比较数量。

    _forward_hooks 是 PyTorch 的内部诊断接口；生产消融代码只使用公开注册/remove API。
    这里不检查全局 hook、编译图或外部线程；验证脚本使用刚离线加载的串行 eager 模型。
    """
    return {
        name: tuple((key, id(callback)) for key, callback in module._forward_hooks.items())
        for name, module in model.named_modules()
    }


def primary(output):
    """从实际观察值独立提取主 Tensor，不调用生产代码的输出替换实现。"""
    if isinstance(output, torch.Tensor):
        return output
    if type(output) in (tuple, list) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"Unexpected observed output type: {type(output).__name__}")


def forward(model, inputs):
    """每次均从完整相同输入开始，显式禁用 KV cache，返回全部输入位置的 logits。"""
    result = model(**inputs, use_cache=False, return_dict=True)
    check(not torch.is_grad_enabled() and not result.logits.requires_grad, "Forward recorded gradients")
    check(torch.isfinite(result.logits).all().item(), "Non-finite logits")
    return result.logits


def run_case(model, layers, layer, component, inputs, baseline, *, inject_failure=False):
    """检查一个独立目标并恢复；非目标比较发生在同一次前向的 hook 前/后。

    上游消融会合法地改变下游输入，故不能要求非目标激活等于未消融基线。这里比较的是
    同一个模块本次返回值在我们的 hook 前后是否相等；目标输出另要求为零且不改写原张量。
    捕获的少量短序列激活仅存在于本函数内，不写入报告。异常测试在目标后立即抛错，
    不要求剩余层执行；正常测试必须观察完整 48 个子模块，并在每次清理后重新验证 logits。
    """
    before_hooks = hook_inventory(model)
    raw, seen, target_description = {}, set(), {}
    target_key = (layer.index, component)
    block_nonzero = []
    caught = False
    state = None

    def capture_before(key):
        """持有原输出引用和小副本，以检出错误的原地清零。"""
        def record(_module, _inputs, output):
            raw[key] = (output, primary(output).clone())
        return record

    def capture_after(key):
        """本回调排在消融 hook 之后，必须看到模型接收到的替代输出。"""
        def record(_module, _inputs, output):
            original, saved = raw[key]
            value = primary(output)
            check(type(output) is type(original), f"Container type changed at {key}")
            check(value.shape == saved.shape and value.dtype == saved.dtype and value.device == saved.device, "Tensor metadata changed")
            check(torch.equal(primary(original), saved), f"Original activation modified in place at {key}")
            if type(output) in (tuple, list):
                check(len(output) == len(original), "Auxiliary output length changed")
                check(all(new is old for new, old in zip(output[1:], original[1:])), "Auxiliary identity changed")
            if key == target_key:
                check(value is not primary(original), "Zero output aliases original Tensor object")
                check(torch.count_nonzero(value).item() == 0, "Target output was not zero")
                target_description.update(
                    before=describe_output(original), after=describe_output(output),
                    original_main_nonzero=torch.count_nonzero(saved).item() > 0,
                )
            else:
                check(torch.equal(value, saved), f"Non-target module directly modified at {key}")
            seen.add(key)
            if inject_failure and key == target_key:
                raise InjectedForwardFailure("injected after zeroed target output")
        return record

    with ExitStack() as observers:
        for item in layers:
            for name, module in (("attention", item.attention_module), ("mlp", item.mlp_module)):
                observers.callback(module.register_forward_hook(capture_before((item.index, name))).remove)
        try:
            with zero_ablation(model, layer, component) as state:
                for item in layers:
                    for name, module in (("attention", item.attention_module), ("mlp", item.mlp_module)):
                        observers.callback(module.register_forward_hook(capture_after((item.index, name))).remove)
                block = model.get_submodule(layer.block_path)
                observers.callback(block.register_forward_hook(
                    lambda _module, _inputs, output: block_nonzero.append(torch.count_nonzero(primary(output)).item() > 0)
                ).remove)
                forward(model, inputs)
        except InjectedForwardFailure:
            if not inject_failure:
                raise
            caught = True
    check(caught == inject_failure, "Injected exception path was not reached")
    check(state is not None and state.calls == 1, "Target must be replaced exactly once per full input")
    check(target_key in seen, "Zeroed output was not observed")
    if not inject_failure:
        check(len(seen) == 2 * len(layers), "Did not observe all Attention/MLP modules")
        check(block_nonzero == [True], "Whole block unexpectedly became zero")
    check(hook_inventory(model) == before_hooks, "Hooks not restored after scope")
    restored = forward(model, inputs)
    check(torch.equal(baseline, restored), "Baseline logits not exactly restored")
    check(all(value.grad is None and not value.requires_grad for value in model.parameters()), "Base parameters acquired gradients")
    return {
        "layer_index": layer.index, "component": component, "module_path": state.module_path,
        "status": "PASS", "successful_replacements": state.calls,
        "target_output": target_description, "observed_submodules": len(seen),
        "target_is_zero": True, "original_activations_unmodified": True,
        "auxiliary_outputs_preserved": True, "non_target_hook_inputs_outputs_equal": True,
        "whole_target_block_nonzero": block_nonzero[0] if block_nonzero else None,
        "injected_forward_failure_caught": caught, "hooks_restored": True,
        "restored_logits_exactly_equal": True, "gradients_absent": True,
    }


def validate(model, tokenizer, text, indices, max_input_tokens):
    """依次验证指定层，首层额外做两条异常路径；不生成 $t^*$、概率差或排名。

    参数冻结、eval 是本验证器明确执行的校准前置操作，不是 zero_ablation 隐式副作用。
    文字是短结构探查输入而非 M2 校准集，不把单输入契约验证声称为完整校准。
    """
    layers = discover_layer_modules(model)
    model.eval().requires_grad_(False)
    inputs = tokenizer(text, return_tensors="pt", padding=False, truncation=False, add_special_tokens=True)
    position = prediction_position(inputs["input_ids"], inputs.get("attention_mask"))
    check(inputs["input_ids"].shape[1] <= max_input_tokens, "Input exceeds smoke-test token limit")
    initial_hooks = hook_inventory(model)
    check(all(not hooks for hooks in initial_hooks.values()), "Validator expects a freshly loaded model without local hooks")
    before_parameters = parameter_fingerprint(model)
    cases = []
    failures = []
    with torch.inference_mode():
        baseline = forward(model, inputs).clone()
        for index in indices:
            for component in ("attention", "mlp"):
                result = run_case(model, layers, layers[index], component, inputs, baseline)
                cases.append(result)
                print(f"PASS layer={index} component={component}: zero, scope and exact recovery", flush=True)
        for component in ("attention", "mlp"):
            failures.append(run_case(model, layers, layers[indices[0]], component, inputs, baseline, inject_failure=True))
            print(f"PASS injected forward exception component={component}: cleanup and exact recovery", flush=True)
    after_parameters = parameter_fingerprint(model)
    check(before_parameters == after_parameters, "Model parameter bytes changed")
    check(hook_inventory(model) == initial_hooks, "Leaked hooks after all cases")
    check(all(not module.training for module in model.modules()), "Model left eval mode")
    return {
        "status": "PASS", "scope": "M3 zero-ablation contract only; not calibration or path patching",
        "model_type": type(model).__name__, "configured_layers": len(layers),
        "resolved_revision": getattr(model.config, "_commit_hash", None),
        "input": {"text": text, "input_ids": inputs["input_ids"][0].tolist(), "prediction_position": position},
        "selected_layer_indices": indices, "normal_cases": cases, "exception_cases": failures,
        "baseline_logits_shape": list(baseline.shape), "logits_dtype": str(baseline.dtype),
        "parameter_fingerprint_before": before_parameters, "parameter_fingerprint_after": after_parameters,
        "all_parameters_frozen": True, "all_gradients_absent": True, "all_modules_eval": True,
        "all_hooks_restored": True, "use_cache": False,
    }


def positive_int(value):
    """在读取权重前限制样本长度/CPU 线程参数。"""
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def pinned_revision(value):
    """只接受完整 SHA，拒绝浮动 main/tag；是否已缓存仍由本地加载器核验。"""
    if re.fullmatch(r"[0-9a-fA-F]{40}", value) is None:
        raise argparse.ArgumentTypeError("revision must be a 40-character hexadecimal SHA")
    return value.lower()


def main():
    """只写新报告；默认一个层，通过后才显式请求 --all-layers，不自动推进 M4。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", type=pinned_revision, default=MODEL_REVISION)
    parser.add_argument("--cache-dir", type=Path, default=MODEL_CACHE)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--layer", type=int, action="append", help="Zero-based layer; repeat for several independent targets")
    group.add_argument("--all-layers", action="store_true", help="Explicitly validate all 24 layers, never jointly")
    parser.add_argument("--text", default="Hello, world!")
    parser.add_argument("--max-input-tokens", type=positive_int, default=64)
    parser.add_argument("--seed", type=int, default=20261114)
    parser.add_argument("--threads", type=positive_int, default=4)
    parser.add_argument("--output", type=Path, required=True, help="New JSON path; existing output is never overwritten")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    indices = list(range(EXPECTED_LAYERS)) if args.all_layers else (args.layer or [0])
    if len(set(indices)) != len(indices) or any(not 0 <= index < EXPECTED_LAYERS for index in indices):
        parser.error("Layers must be unique zero-based indices in 0..23")
    if not args.text.strip():
        parser.error("--text must be non-empty")
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    started = perf_counter()
    model, tokenizer = load_model_and_tokenizer(
        args.model_id, revision=args.revision, cache_dir=args.cache_dir, allow_download=False,
    )
    report = validate(model, tokenizer, args.text, indices, args.max_input_tokens)
    sources = ("src/adaptive_lora/ablation.py", "src/adaptive_lora/model_loader.py", "scripts/validate_ablation.py", "tests/test_ablation.py")
    report.update(
        created_at_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=perf_counter() - started,
        model_id=args.model_id, requested_revision=args.revision, cache_dir=str(args.cache_dir),
        seed=args.seed, cpu_threads=args.threads, command=sys.argv,
        packages={name: metadata.version(name) for name in ("torch", "transformers")},
        git_baseline_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip() or None,
        git_status=subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.splitlines(),
        code_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, ensure_ascii=False)
        output.write("\n")
    print(f"PASS: {len(report['normal_cases'])} normal cases, {len(report['exception_cases'])} injected failures; parameters unchanged")
    print(f"Ablation validation written: {args.output}")


if __name__ == "__main__":
    main()

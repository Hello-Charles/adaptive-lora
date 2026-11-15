"""M6 的共享 prompt-only 贪心生成与独立 exact-match smoke 评分接口。

生成器不接收 reference/labels，输入只能是无 padding 的 user prefix，再逐步追加自身预测。
评分器在生成后按样本 ID 对齐参考答案；strip-only、短生成上限均是复现工程决策，不是
论文 commonsense 答案抽取或 math chain-of-thought/regex 评估的实现。
"""

import torch

from adaptive_lora.protocol import stable_hash


def _requests(model, tokenizer, prompts, max_new_tokens):
    """第一次前向前验证全部有界单样本请求，拒绝 padding、答案字段和非法 token。"""
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 8:
        raise ValueError("M6 smoke generation requires 1–8 new tokens")
    vocab, eos = getattr(model.config, "vocab_size", None), tokenizer.eos_token_id
    if type(vocab) is not int or vocab < 2 or type(eos) is not int or not 0 <= eos < vocab:
        raise ValueError("Model vocab and tokenizer EOS must be explicit and compatible")
    prompts = list(prompts)
    if not 1 <= len(prompts) <= 3:
        raise ValueError("M6 smoke accepts 1–3 evaluation prompts")
    requests, identifiers = [], set()
    for row in prompts:
        if set(row) != {"sample_id", "input_ids", "attention_mask", "prompt_sha256"}:
            raise ValueError("Generation requests must contain only prefix inputs and identity")
        identifier, ids, mask = (row[key] for key in ("sample_id", "input_ids", "attention_mask"))
        if (not isinstance(identifier, str) or not identifier or identifier in identifiers
                or not isinstance(ids, list) or not 1 <= len(ids) <= 96
                or any(type(token) is not int or not 0 <= token < vocab for token in ids)
                or not isinstance(mask, list) or any(type(value) is not int for value in mask) or mask != [1] * len(ids)):
            raise ValueError("Invalid single-example generation IDs/mask or repeated sample ID")
        identifiers.add(identifier)
        requests.append({"sample_id": identifier, "input_ids": list(ids), "prefix_token_sha256": stable_hash(ids)})
    if any(parameter.device.type != "cpu" or parameter.grad is not None for parameter in model.parameters()):
        raise ValueError("Evaluation requires CPU parameters without pre-existing gradients")
    if any(module._forward_hooks or module._forward_pre_hooks for module in model.modules()):
        raise ValueError("Evaluation rejects existing local forward hooks, including ablation hooks")
    return requests, eos


def predict_greedy(model, tokenizer, prompts, *, max_new_tokens=4):
    """返回逐样本预测文本/token/停止原因；不读取 gold answer，不改 requires_grad 或参数。

    使用完整未缓存的 [1,S] CPU 输入、全一 mask；因此 logits[0,-1] 确为最后有效 token 的
    next-token 分布，不把 right padding 当预测位置。直接 argmax 实现 temperature=0 的贪心
    决策，不对零温度做除法或采样；EOS 只有显式 tokenizer.eos_token_id，特殊解码参数固定。

    所有子模块临时 eval 且 no_grad。正常/异常均逐模块恢复原 training flag，以保留调用者
    原本混合的模式；不支持自定义 train() 的额外副作用。不会清空调用者已有梯度，而是在
    开始前拒绝这些状态。基础/adapter 字节一致性由比较 runner 在评估前后显式复核。
    """
    requests, eos = _requests(model, tokenizer, prompts, max_new_tokens)
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    predictions = []
    try:
        with torch.no_grad():
            for request in requests:
                ids = torch.tensor([request["input_ids"]], dtype=torch.long)
                generated, reason = [], "length_limit"
                for _ in range(max_new_tokens):
                    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
                    if (not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (1, ids.shape[1], model.config.vocab_size)
                            or not torch.isfinite(logits[0, -1]).all().item()):
                        raise ValueError("Invalid or nonfinite final-position generation logits")
                    token = int(logits[0, -1].argmax().item())
                    generated.append(token)
                    if token == eos:
                        reason = "eos"
                        break
                    ids = torch.cat((ids, torch.tensor([[token]], dtype=torch.long)), dim=1)
                text = tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                if not isinstance(text, str):
                    raise ValueError("Tokenizer must decode predictions to text")
                predictions.append({"sample_id": request["sample_id"], "prediction": text,
                                    "generated_token_ids": generated, "finish_reason": reason,
                                    "forward_calls": len(generated), "prompt_token_count": len(request["input_ids"]),
                                    "prefix_token_sha256": request["prefix_token_sha256"]})
    finally:
        for module, training in modes:
            module.training = training
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError("Unexpected gradients after evaluation")
    return predictions


def score_exact_match(predictions, references):
    """生成完成后按唯一 ID 对齐评分，返回逐项对照和正确数/总数/比例，不丢弃不匹配样本。

    [复现决策] 仅 strip 两端空白，大小写和内部空白保持原样。reference 不出现在任何模型
    输入路径；相同参考文本可以对应不同问题，不能据此推断泄漏。空预测计错误，不跳过。
    """
    predictions, references = list(predictions), list(references)
    if not predictions or len(predictions) != len(references):
        raise ValueError("Predictions and references must be non-empty and have equal counts")
    def indexed(rows, field):
        """要求 ID 非空唯一、内容为文本，不能用 dict 覆盖重复 ID 后悄悄缩小分母。"""
        result = {}
        for row in rows:
            identifier, text = row["sample_id"], row[field]
            if not isinstance(identifier, str) or not identifier or identifier in result or not isinstance(text, str):
                raise ValueError("Invalid or duplicate prediction/reference ID or text")
            result[identifier] = text
        return result
    predicted, gold = indexed(predictions, "prediction"), indexed(references, "text")
    if set(predicted) != set(gold):
        raise ValueError("Prediction and reference sample IDs differ")
    items = [{"sample_id": identifier, "prediction": text, "reference": gold[identifier],
              "correct": text.strip() == gold[identifier].strip()} for identifier, text in predicted.items()]
    correct = sum(item["correct"] for item in items)
    return {"metric": "smoke_exact_match_strip_only_not_paper_accuracy", "correct": correct,
            "total": len(items), "accuracy": correct / len(items), "items": items}

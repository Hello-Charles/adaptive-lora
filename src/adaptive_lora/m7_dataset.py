"""固定LLM-Adapters文件的离线字节/字段/抽样及文本输入隔离审计，不下载或分词。

CLI负责先验证已确认M7配置；本模块也可接受小型合成dataset规格供测试。调用者需先在
独立目录获取固定原文件并确认许可，避免并发修改。本文本门禁不认证来源、审查每条内容
权利或证明token/语义隔离；真实tokenizer、截断与模型结构仍须独立验收。
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import sys

from adaptive_lora.m7_protocol import TASK_LABELS, build_commonsense_prompt


def canonical_sha256(value):
    """规范JSON哈希用于记录内容身份；全行哈希可含答案，不是模型输入或脱敏机制。"""
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _pairs(pairs):
    """拒绝重复JSON键，避免同一来源存在多个互相覆盖的指令或参考答案。"""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def _nonfinite(value):
    """标准JSON不接受NaN/Infinity，不能让非有限数参与可重复哈希。"""
    raise ValueError("Non-finite JSON constant: " + value)


def source_path(root, relative):
    """只接受根目录下规范POSIX相对文件名，拒绝路径穿越、Windows驱动和symlink。

    这不是并发文件系统沙箱；调用者需确保验证期间没有其他进程替换路径。
    """
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise ValueError("Unsafe source path")
    parts = relative.split("/")
    if PurePosixPath(relative).is_absolute() or any(part in ("", ".", "..") for part in parts):
        raise ValueError("Unsafe source path")
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Source root must be a real directory")
    path = root
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("Symlink source path")
    return path


def read_verified_rows(root, spec):
    """先核对精确长度/Git blob，再解析严格UTF-8 JSON；返回原行顺序及字节来源记录。

    Git blob是SHA1(blob头+原始字节)，与普通SHA256不同。最大100MB只覆盖已确认的单个
    训练文件，防止调用方用超大规格无界读入；大小/hash匹配不是漏洞或来源独立认证。
    """
    size = spec["size_bytes"]
    if type(size) is not int or not 0 < size <= 100_000_000:
        raise ValueError("Invalid bounded source size")
    path = source_path(root, spec["path"])
    if not path.is_file() or path.stat().st_size != size:
        raise ValueError("Source byte length mismatch: " + spec["path"])
    with path.open("rb") as handle:
        raw = handle.read(size + 1)
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()
    if len(raw) != size or blob != spec["git_blob_sha1"]:
        raise ValueError("Source Git blob mismatch: " + spec["path"])
    rows = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_nonfinite)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Source must contain a non-empty JSON list")
    return rows, {"path": spec["path"], "size_bytes": len(raw), "git_blob_sha1": blob,
                  "sha256": hashlib.sha256(raw).hexdigest(), "row_count": len(rows)}


def validate_rows(rows, task=None):
    """审查每行instruction/input类型及参考字段，不把answer/output拼入prompt。

    训练使用非空output；评估使用来源answer且要求精确规范标签，不猜测yes/no映射。
    额外字段仅统计键集合，不执行或静默丢弃；原行的完整哈希仍保留这些字段。
    缺失input按已确认模板的空输入处理，None也由原模板规范为空输入。
    """
    if task is not None and task not in TASK_LABELS:
        raise ValueError("Unknown evaluation task")
    schemas = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Row {index} must be an object")
        build_commonsense_prompt(row.get("instruction"), row.get("input", ""))
        field = "output" if task is None else "answer"
        answer = row.get(field)
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"Row {index} requires non-empty {field}")
        if task is not None and answer not in TASK_LABELS[task]:
            raise ValueError(f"Row {index} has non-canonical reference label for {task}")
        schema = ",".join(sorted(row))
        schemas[schema] = schemas.get(schema, 0) + 1
    return schemas


def sample_indices(population, count, seed):
    """局部Python RNG均匀不放回、保留抽取顺序；Python版本也是复现条件。

    不接受bool、不静默降低数量，且不改变全局random状态。seed来自已确认复现决策，
    不能把它表述为论文公开的逐样本索引或校准抽样强制seed。
    """
    if any(type(value) is not int for value in (population, count, seed)):
        raise ValueError("Sampling values must be ordinary integers")
    if population < 1 or not 0 < count <= population or not 0 <= seed < 2**32:
        raise ValueError("Invalid sampling population/count/seed")
    return random.Random(seed).sample(range(population), count)


def selected_records(rows, indices, path, revision):
    """记录来源位置ID、原行与user-only prompt身份，不保存原文或把参考标签传入模板。

    ID按revision/来源路径/零基行号构造，不是原benchmark官方ID；跨来源ID不同本身
    不证明无内容重叠，因此还必须比较规范prompt字节哈希。
    """
    records = []
    for index in indices:
        row = rows[index]
        instruction, input_text = row["instruction"], row.get("input", "")
        prompt = build_commonsense_prompt(instruction, input_text)
        records.append({"sample_id": f"{revision}:{path}:{index}", "source_index": index,
                        "row_sha256": canonical_sha256(row),
                        "input_fields_sha256": canonical_sha256({"instruction": instruction, "input": input_text}),
                        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()})
    return records


def overlap_summary(train, calibration, evaluations):
    """统计唯一ID/prompt哈希交集及组内重复，不过滤重叠或重新抽样。

    校准来自训练子集是允许的；只把训练/校准对各评估任务的直接文本交集作为阻塞。
    哈希交集是唯一身份数而非样本对数，不检查语义改写、题面内嵌答案或预训练污染。
    """
    def identities(records, key):
        return {row[key] for row in records}

    details = {}
    blocked = False
    for task, records in evaluations.items():
        detail = {}
        for split, source in (("train", train), ("calibration", calibration)):
            for key in ("sample_id", "prompt_sha256"):
                common = sorted(identities(source, key) & identities(records, key))
                detail[f"{split}_{key}_intersection"] = common
                blocked = blocked or bool(common)
        details[task] = detail
    groups = {"train": train, "calibration": calibration, **evaluations}
    duplicates = {name: len(rows) - len(identities(rows, "prompt_sha256")) for name, rows in groups.items()}
    return {"text_disjoint": not blocked, "per_task": details, "within_split_duplicate_prompt_rows": duplicates,
            "calibration_is_training_subset": identities(calibration, "sample_id").issubset(identities(train, "sample_id")),
            "token_identity_checked": False, "semantic_deduplication_checked": False}


def audit_dataset(dataset, root):
    """生成固定抽样清单和纯文本门禁结果，不下载、分词、加载权重或运行训练。

    dataset应来自CLI验证过的M7配置；测试可传小型等价规格。先核验所有源文件和全行
    字段，再按既定seed抽样。发现重叠只返回BLOCKED_TEXT_OVERLAP并保留原索引；不得
    自动修改协议。即使文本门禁通过，仍明确tokenizer与真实数据内容权利未全面验收。
    """
    rows, provenance = read_verified_rows(root, dataset["train_file"])
    provenance["field_schemas"] = validate_rows(rows)
    train_indices = sample_indices(len(rows), dataset["train_samples"], dataset["train_sampling_seed"])
    cal_positions = sample_indices(len(train_indices), dataset["calibration_samples"], dataset["calibration_sampling_seed"])
    cal_indices = [train_indices[position] for position in cal_positions]
    train = selected_records(rows, train_indices, provenance["path"], dataset["revision"])
    by_index = {row["source_index"]: row for row in train}
    calibration = [dict(by_index[index]) for index in cal_indices]
    sources = [provenance]
    evaluations, eval_indices = {}, {}
    for spec in dataset["evaluation_files"]:
        task = spec["task"]
        if task in evaluations:
            raise ValueError("Duplicate evaluation task")
        eval_rows, info = read_verified_rows(root, spec)
        info["field_schemas"] = validate_rows(eval_rows, task)
        info["task"] = task
        indices = sample_indices(len(eval_rows), spec["sample_count"], dataset["evaluation_sampling_seed"])
        sources.append(info)
        evaluations[task] = selected_records(eval_rows, indices, spec["path"], dataset["revision"])
        eval_indices[task] = indices
    overlap = overlap_summary(train, calibration, evaluations)
    protocol = {"dataset_revision": dataset["revision"], "train_indices": train_indices,
                "calibration_training_positions": cal_positions, "calibration_source_indices": cal_indices,
                "evaluation_indices": eval_indices, "python_version": sys.version.split()[0],
                "seeds": {key: dataset[key] for key in ("train_sampling_seed", "calibration_sampling_seed", "evaluation_sampling_seed")}}
    return {"schema_version": 1, "status": "TEXT_GATE_PASS_TOKENIZER_PENDING" if overlap["text_disjoint"] else "BLOCKED_TEXT_OVERLAP",
            "sources": sources, "sampling": protocol, "sampling_sha256": canonical_sha256(protocol),
            "train": train, "calibration": calibration, "evaluation": evaluations, "overlap": overlap,
            "tokenizer_validated": False, "content_rights_individually_cleared": False,
            "real_model_structure_validated": False, "training_started": False}

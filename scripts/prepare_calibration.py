"""准备 M2 校准清单；只下载固定 parquet，不加载模型权重、不执行远端数据代码。

默认离线；显式 --allow-download 只允许数据文件下载，tokenizer 始终复用本地缓存。
下载目录必须全新且与代码分开；从项目目录以 python -I 运行，不从下载目录启动解释器。
"""

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path, PurePosixPath
import platform
import re
import subprocess
import sys
from urllib.parse import urlsplit

from adaptive_lora.data import (
    DATASET_ID,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_ID,
    DEFAULT_N,
    DEFAULT_SEED,
    build_manifest,
    sample_indices,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_REVISION_DEFAULT = "060db6499f32faf8b98477b0a26969ef7d8b9987"
MODEL_CACHE_DEFAULT = "data/models/qwen2_5_0_5b_060db6499f32faf8b98477b0a26969ef7d8b9987"
# [复现决策] 2026-11-14 查询 Alpaca Hub 元数据后固定的 train parquet 身份；不是论文数据版本。
DATASET_REVISION_DEFAULT = "dce01c9b08f87459cf36a430d809084718273017"
DATASET_FILE_DEFAULT = "data/train-00000-of-00001-a09b74b3ef9c3b56.parquet"
DATASET_SHA256_DEFAULT = "06391b656a06fd3fb9d213160ef2398796c3b7f3dc75ef1e3ced30d461517073"


def file_sha256(path):
    """流式计算文件原始字节哈希；不解析或执行内容，也不把大型文件整体载入内存。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(*, dataset_id, revision, cache_dir, filename, expected_sha256, allow_download, endpoint):
    """返回固定 train parquet 的可索引 Dataset 和文件来源指纹，不复制全量行到 Python。

    Hub 仅负责取一个固定 SHA 下的文件；local_files_only 明确限制离线路径，不依赖
    setdefault 环境变量或模块导入顺序。先验证字节 SHA-256 再用已安装的 parquet reader
    解析，不请求仓库数据脚本、不执行下载目录内代码、不产生 datasets 的长路径缓存。
    parquet 整表会解码到 Arrow 内存，适用于本次约 24 MB 的 smoke 文件，不适合任意
    大数据集。公开访问 token=False，不向可选镜像发送本机 Hub 凭据。
    """
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(
        repo_id=dataset_id, repo_type="dataset", filename=filename, revision=revision,
        cache_dir=str(cache_dir / "hub"), endpoint=endpoint,
        local_files_only=not allow_download, token=False,
    ))
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"Dataset SHA-256 mismatch: expected {expected_sha256}, found {actual}")

    from datasets import Dataset
    from pyarrow.parquet import read_table

    # 直接解析本地文件，不经过会按绝对路径生成锁文件的 datasets 下载/处理缓存。
    # memory_map 仅影响源文件访问；parquet 列解压仍消耗内存，不能当作流式读取保证。
    rows = Dataset(read_table(str(path), memory_map=True))
    return rows, {"file": filename, "sha256": actual, "size_bytes": path.stat().st_size}


def load_tokenizer(model_id, revision, cache_dir):
    """只读固定 revision 的本地 tokenizer；数据下载开关不能扩大为模型联网许可。

    不加载模型权重，不启用远端代码。模板是否与分词协议一致由测试及真实清单验收核对。
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_id, revision=revision, cache_dir=str(cache_dir), use_fast=True,
        local_files_only=True, trust_remote_code=False, token=False,
    )


def positive_int(value):
    """argparse 在任何导入、目录创建和下载前拒绝非正长度/样本数。"""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def hex_digest(value, length):
    """限制为完整十六进制 SHA，拒绝 main/tag 等可变 revision；不声称验证远端存在性。"""
    if re.fullmatch(rf"[0-9a-fA-F]{{{length}}}", value) is None:
        raise argparse.ArgumentTypeError(f"Expected a {length}-character hexadecimal SHA")
    return value.lower()


def provenance(args):
    """保存本机运行证据；HEAD 只表示基线，dirty 状态和源码哈希补足未提交代码身份。

    时间戳、缓存绝对路径、Git 状态不参与内容指纹；同样本配置的跨路径复跑仍可比较。
    命令参数不接受认证 token，避免把凭据写入清单；完整输入清单仍不可未经审查发布。
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False,
        ).stdout.strip() or None
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=False,
        ).stdout.splitlines()
    except FileNotFoundError:
        commit, status = None, None
    sources = (
        "src/adaptive_lora/data.py", "scripts/prepare_calibration.py",
        "tests/test_data.py", "tests/test_prepare_calibration.py",
    )
    packages = ("torch", "transformers", "tokenizers", "peft", "datasets", "pyarrow", "huggingface-hub")
    return {
        "python": platform.python_version(),
        "packages": {name: metadata.version(name) for name in packages},
        "git": {"commit": commit, "dirty": bool(status) if status is not None else None, "status": status},
        "code_sha256": {name: file_sha256(ROOT / name) for name in sources},
        "command": sys.argv,
        "dataset_cache_dir": str(args.dataset_cache_dir.resolve()),
        "model_cache_dir": str(args.model_cache_dir.resolve()),
        "dataset_transport_endpoint": args.hub_endpoint,
        "dataset_download_allowed": args.allow_download,
        "tokenizer_local_files_only": True,
    }


def main():
    """写一个新 manifest，失败不留下成功清单；不覆盖历史结果或复用旧下载目录。

    [复现决策] 默认只支持已核验的 Alpaca 单文件 train 导出；替换数据时调用者必须
    同时给出文件名、固定 revision 及 SHA-256，并保证含文本 instruction/input 字段。
    下载失败的目录予以保留作为诊断证据；重试须另给新目录，不自动删除缓存。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--dataset-revision", type=lambda value: hex_digest(value, 40), default=DATASET_REVISION_DEFAULT)
    parser.add_argument("--dataset-file", default=DATASET_FILE_DEFAULT, help="Pinned train parquet path in the dataset repo")
    parser.add_argument("--dataset-sha256", type=lambda value: hex_digest(value, 64), default=DATASET_SHA256_DEFAULT)
    parser.add_argument("--dataset-cache-dir", type=Path, required=True, help="Dedicated cache: new for download, existing offline")
    parser.add_argument("--hub-endpoint", default="https://huggingface.co", help="HTTPS transport endpoint; never receives local Hub credentials")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision", type=lambda value: hex_digest(value, 40), default=MODEL_REVISION_DEFAULT)
    parser.add_argument("--model-cache-dir", type=Path, default=ROOT / MODEL_CACHE_DEFAULT)
    parser.add_argument("--n", type=positive_int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-length", type=positive_int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--output", type=Path, required=True, help="New manifest path (never overwritten)")
    parser.add_argument("--allow-download", action="store_true", help="Allow only the pinned dataset parquet download")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    filename = PurePosixPath(args.dataset_file)
    if filename.is_absolute() or ".." in filename.parts or filename.suffix != ".parquet":
        parser.error("--dataset-file must be a relative parquet path without '..'")
    if urlsplit(args.hub_endpoint).scheme != "https":
        parser.error("--hub-endpoint must use HTTPS")
    if args.allow_download:
        if args.dataset_cache_dir.exists():
            parser.error(f"Download cache must be a new, empty directory: {args.dataset_cache_dir}")
    elif not args.dataset_cache_dir.is_dir():
        parser.error(f"Offline cache does not exist: {args.dataset_cache_dir}")

    try:
        # 先确认本地 tokenizer 可用，避免缺缓存时无意义下载数据；方向不被脚本偷偷改写。
        tokenizer = load_tokenizer(args.model_id, args.model_revision, args.model_cache_dir)
        if args.allow_download:
            args.dataset_cache_dir.mkdir(parents=True, exist_ok=False)
        rows, source = load_rows(
            dataset_id=args.dataset_id, revision=args.dataset_revision, cache_dir=args.dataset_cache_dir,
            filename=args.dataset_file, expected_sha256=args.dataset_sha256,
            allow_download=args.allow_download, endpoint=args.hub_endpoint,
        )
        indices = sample_indices(len(rows), args.n, args.seed)
        manifest = build_manifest(
            rows=rows, indices=indices, tokenizer=tokenizer, dataset_id=args.dataset_id,
            dataset_revision=args.dataset_revision, model_id=args.model_id, model_revision=args.model_revision,
            max_length=args.max_length, seed=args.seed, dataset_source=source,
        )
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Calibration preparation failed: {exc}\n")
    manifest["provenance"] = provenance(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # 独占创建防止预检查与写入之间的竞争覆盖；完整 prompt 默认放 results/ 等 Git 忽略路径。
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2, ensure_ascii=False)
        output.write("\n")
    counts = [sample["token_count"] for sample in manifest["samples"]]
    print(f"Calibration manifest written: {args.output}")
    print(f"dataset rows: {len(rows)}; sampled: {len(indices)}; seed: {args.seed}")
    print(f"token counts: {min(counts)}..{max(counts)}; content SHA-256: {manifest['content_sha256']}")


if __name__ == "__main__":
    main()

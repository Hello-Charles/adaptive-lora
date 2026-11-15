"""离线审计已获取的固定M7数据文件，不联网、不分词、不加载模型或训练。

从本脚本所属已审阅仓库加载标准库模块，不使用输入目录作为模块搜索路径。调用者负责
合法来源、下载批准与新结果目录；运行保留完整索引/哈希但不把它当作完整M7验收标记。
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from adaptive_lora.m7_dataset import audit_dataset
from adaptive_lora.m7_protocol import load_m7_config, protocol_sha256


def write_new(path, value):
    """独占创建JSON，已存在的输入、结果或失败记录均不能被覆盖。"""
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def source_hashes():
    """哈希覆盖实际加载的代码和固定配置，不把Git HEAD误称为含dirty改动的完整来源。"""
    paths = ("scripts/audit_m7_dataset.py", "src/adaptive_lora/m7_dataset.py",
             "src/adaptive_lora/m7_protocol.py", "src/adaptive_lora/__init__.py")
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths}


def git_state():
    """只读记录基线与dirty，不提交/push；不可用时明确未知，不构造成功SHA。"""
    try:
        def query(*args):
            return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True, timeout=5).strip()
        return {"commit": query("rev-parse", "HEAD"), "dirty": bool(query("status", "--porcelain"))}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def main(argv=None):
    """返回0=文本门禁通过但tokenizer待验，2=重叠阻塞或输入/IO错误。

    输入错误只保存error.json，不造成功summary；重叠则保留完整原抽样audit.json与
    BLOCKED summary，绝不去重、重抽或缩小评估分母。输出目录已存在时不写任何文件。
    """
    parser = argparse.ArgumentParser(description="Offline fixed M7 JSON integrity/sampling/text gate; no download or tokenizer")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/qwen2_5_1_5b_commonsense_r16.json")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    created = False
    try:
        config = load_m7_config(args.config)
        if any(item["status"] != "confirmed" for item in config["decisions"]):
            raise ValueError("Protocol decisions are not all confirmed")
        before = source_hashes()
        args.output_dir.mkdir(parents=True, exist_ok=False)
        created = True
        audit = audit_dataset(config["dataset"], args.input_root)
        if source_hashes() != before:
            raise ValueError("Audit source changed during run")
        # 再复查每个原JSON普通字节，检测审计期间的改动；不声称并发安全或独立来源认证。
        for item in audit["sources"]:
            path = args.input_root / item["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError("Dataset source changed during audit")
        audit.update({"config_sha256": protocol_sha256(config), "source_files_sha256": before,
                      "git": git_state(), "recorded_at_utc": datetime.now(timezone.utc).isoformat()})
        write_new(args.output_dir / "audit.json", audit)
        overlap_counts = {task: {name: len(values) for name, values in detail.items()}
                          for task, detail in audit["overlap"]["per_task"].items()}
        summary = {"status": audit["status"], "sources": audit["sources"],
                   "train_count": len(audit["train"]), "calibration_count": len(audit["calibration"]),
                   "evaluation_counts": {task: len(rows) for task, rows in audit["evaluation"].items()},
                   "sampling_sha256": audit["sampling_sha256"], "config_sha256": audit["config_sha256"],
                   "overlap_counts": overlap_counts,
                   "within_split_duplicate_prompt_rows": audit["overlap"]["within_split_duplicate_prompt_rows"],
                   "tokenizer_validated": False, "real_model_structure_validated": False,
                   "content_rights_individually_cleared": False, "training_started": False,
                   "git": audit["git"], "source_files_sha256": before,
                   "audit_file_sha256": hashlib.sha256((args.output_dir / "audit.json").read_bytes()).hexdigest()}
        write_new(args.output_dir / "summary.json", summary)
    except (OSError, ValueError, UnicodeError, KeyError) as error:
        if created:
            write_new(args.output_dir / "error.json", {"status": "ERROR", "error_type": type(error).__name__, "error": str(error)})
        print("ERROR: " + str(error), file=sys.stderr)
        return 2
    print(summary["status"] + ": text-only dataset gate; tokenizer/model/training not validated")
    return 0 if audit["overlap"]["text_disjoint"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

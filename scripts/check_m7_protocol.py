"""离线检查M7候选协议并存档本地摘要，不执行部署、安装、下载或训练。

应从已审查的项目检出运行，不能复制到模型/数据下载目录执行。为使尚未安装依赖的
服务器也能使用，只把本脚本所属仓库的src加入搜索路径，不加入配置目录或当前目录。
-I -S亦可运行；confirmed配置只改变审查状态，不能授权费用或证明GPU运行已通过。
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = str(ROOT / "src")
if SOURCE not in sys.path:
    sys.path.insert(0, SOURCE)

from adaptive_lora.m7_protocol import build_preflight_report, load_m7_config

DEFAULT_CONFIG = ROOT / "configs/qwen2_5_1_5b_commonsense_r16.json"


def source_hashes():
    """记录本检查器使用的固定源码/候选依赖字节，不读取模型、数据或凭据。

    配置规范内容哈希由协议模块计算；这些文件的实际字节另记，不能将两种哈希混淆。
    调用者须避免并行修改源码；运行末尾复查只是变更检测，不是并发文件系统快照。
    """
    paths = ("src/adaptive_lora/__init__.py", "src/adaptive_lora/m7_protocol.py", "scripts/check_m7_protocol.py", "requirements-linux-cu124-m7.in")
    return {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths}


def git_provenance():
    """尽力只读获取本地Git基线与dirty状态；Git不可用时明确记录未知，不伪造版本。"""
    try:
        def query(*arguments):
            """仅运行固定Git只读参数，配置内容从不拼入命令。"""
            return subprocess.check_output(["git", "-C", str(ROOT), *arguments], stderr=subprocess.DEVNULL, text=True, timeout=5).strip()

        return {"status": "recorded", "baseline_commit": query("rev-parse", "HEAD"), "branch": query("rev-parse", "--abbrev-ref", "HEAD"), "dirty": bool(query("status", "--porcelain"))}
    except (OSError, subprocess.SubprocessError):
        return {"status": "unavailable", "baseline_commit": None, "branch": None, "dirty": None}


def write_new_json(path, payload):
    """只写新建的本地文件；上游必须已创建独占输出目录，失败保留部分证据。"""
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def main(argv=None):
    """返回0=配置记录已确认但不可执行，2=待确认或输入/IO错误。

    先验证配置再创建输出目录；旧目录一律拒绝，不删除/覆盖。config.json为实际采用的
    配置快照，preflight.json最后写入，明确标注审查状态；它不是训练成功标记。若写入
    中断，可能只留下配置，不应据此认定预检完成。没有任何远程或付费作业入口。
    """
    parser = argparse.ArgumentParser(description="Offline M7 protocol review only; no download/install/training")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Pinned candidate JSON; never executable code")
    parser.add_argument("--output-dir", type=Path, required=True, help="New local review directory; existing paths are refused")
    args = parser.parse_args(argv)
    try:
        config = load_m7_config(args.config)
        report = build_preflight_report(config)
        before = source_hashes()
        report["source_files_sha256"] = before
        report["git"] = git_provenance()
        report["scope"] = "offline protocol review only; no framework imports, remote commands or runtime validation"
        args.output_dir.mkdir(parents=True, exist_ok=False)
        write_new_json(args.output_dir / "config.json", config)
        if source_hashes() != before:
            raise ValueError("Checker source changed during review; no completion report written")
        write_new_json(args.output_dir / "preflight.json", report)
    except (OSError, ValueError, UnicodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(report["review_status"] + ": configuration review only; execution_authorized=false; runtime_validated=false")
    if report["pending_decisions"]:
        print("Pending decisions: " + ", ".join(report["pending_decisions"]))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

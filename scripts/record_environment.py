"""环境快照只反映当前解释器，不安装依赖、不主动加载模型。

默认检查会执行已安装包的导入代码；仅查版本时须使用 --metadata-only。
元数据存在、包可导入和模型能完成前向是不同层级的证据，不能互相替代。
"""

import argparse
from datetime import datetime, timezone
import importlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = (
    "torch",
    "transformers",
    "peft",
    "datasets",
    "accelerate",
    "pytest",
    "setuptools",
    "wheel",
    "pip",
)


def package_status(name, verify_imports=True):
    """区分未安装、未检查、导入失败和导入成功，不把版本号当作可用性证明。

    name 同时用作发行包名和导入名，仅适用于两者一致的 PACKAGES 项。
    实际导入可能有包自身的副作用；捕获常规异常是为了保留其余包的环境证据。
    """
    try:
        version = metadata.version(name)
    except metadata.PackageNotFoundError:
        return {"status": "not_installed", "version": None}

    if not verify_imports:
        return {"status": "not_checked", "version": version}

    try:
        importlib.import_module(name)
    except Exception as exc:
        return {"status": "import_failed", "version": version, "error": f"{type(exc).__name__}: {exc}"}
    return {"status": "ok", "version": version}


def git_status():
    """只把项目根目录自身的仓库 HEAD 作为版本证据，不借用父目录仓库。

    commit 不包含未提交的改动；缺失仓库、Git 或首个提交时以 None 记录，不能伪造 SHA。
    """
    try:
        repository = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return {"status": "git_unavailable", "commit": None}

    if repository.returncode != 0 or Path(repository.stdout.strip()).resolve() != ROOT:
        return {"status": "not_repository", "commit": None}

    commit = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if commit.returncode != 0:
        return {"status": "no_commits", "commit": None}
    return {"status": "ok", "commit": commit.stdout.strip()}


def snapshot(verify_imports=True):
    """记录 UTC 时间及当前进程环境；不同时间的快照不要求逐字节相同。

    包导入可能改变同一进程的状态；本结果不是逐包隔离导入的诊断报告。
    """
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "platform": platform.platform(),
        "packages": {name: package_status(name, verify_imports=verify_imports) for name in PACKAGES},
        "git": git_status(),
    }


def main():
    """仅写入新的 JSON；现有快照是历史证据，即使本次环境变化也不能覆盖。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New JSON file for the snapshot")
    parser.add_argument("--metadata-only", action="store_true", help="Do not import installed packages")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(snapshot(verify_imports=not args.metadata_only), output, indent=2, ensure_ascii=False)
            output.write("\n")
    except FileExistsError:
        parser.error(f"Output already exists: {args.output}")
    print(f"Environment snapshot written: {args.output}")


if __name__ == "__main__":
    main()

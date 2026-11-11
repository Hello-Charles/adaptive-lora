"""Capture the active M0 environment without installing or loading a model."""

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
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "platform": platform.platform(),
        "packages": {name: package_status(name, verify_imports=verify_imports) for name in PACKAGES},
        "git": git_status(),
    }


def main():
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

"""Check tracked working files and all reachable Git history without printing contents."""
import json
import re
import subprocess
from pathlib import Path


def git(*args):
    return subprocess.check_output(["git", *args])


ALLOWED = {".md", ".py", ".json", ".csv", ".yml", ".yaml", ".toml", ".txt", ".ipynb"}
ALLOWED_CSV = {"experiments/results.csv", "logs/submissions.csv"}
SECRET = re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)")
BLOCKED_PARTS = {"train", "data", "samples", "assets", "labels", "images", "bands", "models", "checkpoints", "private", "local", "outputs", "cache"}


def inspect(name, body):
    path = Path(name)
    errors = []
    if set(path.parts) & BLOCKED_PARTS or path.name in {"sites.json", ".env"} or path.name.startswith(".env.") or name == "configs/local.json":
        errors.append("restricted path")
    if path.suffix not in ALLOWED and path.name not in {".gitignore", ".gitkeep", "pre-push"}:
        errors.append("unreviewed file type")
    if path.suffix == ".csv" and name not in ALLOWED_CSV:
        errors.append("unreviewed CSV")
    if len(body) > 1_000_000 or b"\0" in body:
        errors.append("large or binary content")
    if SECRET.search(body):
        errors.append("credential pattern")
    if path.suffix == ".ipynb":
        try:
            cells = json.loads(body).get("cells", [])
            if any(c.get("outputs") or c.get("execution_count") is not None or c.get("attachments") for c in cells):
                errors.append("notebook output/attachments")
        except (ValueError, TypeError):
            errors.append("invalid notebook")
    return errors


def main():
    failures = []
    seen = set()
    # Scan every blob occurrence by filename in all reachable commits.
    for commit in git("rev-list", "--all").decode().splitlines():
        for entry in git("ls-tree", "-r", "-z", commit).split(b"\0"):
            if not entry:
                continue
            meta, raw_name = entry.split(b"\t", 1)
            mode, kind, oid = meta.decode().split()
            name = raw_name.decode()
            key = (oid, name)
            if key in seen:
                continue
            seen.add(key)
            if kind != "blob" or mode == "120000":
                failures.append((name, ["submodule or symlink"]))
                continue
            issues = inspect(name, git("cat-file", "blob", oid))
            if issues:
                failures.append((name, issues))
    for raw_name in git("ls-files", "-z").split(b"\0"):
        if not raw_name:
            continue
        name = raw_name.decode()
        path = Path(name)
        if path.is_symlink():
            failures.append((name, ["symlink"]))
        elif path.is_file():
            issues = inspect(name, path.read_bytes())
            if issues:
                failures.append((name, issues))
    for name, issues in failures:
        print(f"FAIL {name}: {', '.join(issues)}")
    if failures:
        return 1
    print(f"PASS: tracked working files and {len(seen)} historical file versions checked; manual content/license review still required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

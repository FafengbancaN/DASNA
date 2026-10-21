"""Check the files that would be shared from this Git repository."""

import ast
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT, check=True, capture_output=True,
    )
    paths = sorted(set(p for p in result.stdout.decode("utf-8").split("\0") if p))
    errors = []
    total = 0
    python_count = 0
    for name in paths:
        path = ROOT / name
        if not path.is_file():
            errors.append(f"Missing or non-file entry: {name}")
            continue
        size = path.stat().st_size
        total += size
        if size >= 50 * 1024 * 1024:
            errors.append(f"Large release file ({size / 1024**2:.1f} MiB): {name}")
        if path.suffix.lower() in {".mat", ".pickle", ".npz", ".pt", ".pth", ".ckpt", ".pyc"}:
            errors.append(f"Dataset, checkpoint, or cache in release: {name}")
        if ".git" in path.relative_to(ROOT).parts or "__pycache__" in path.parts:
            errors.append(f"Nested Git metadata or Python cache: {name}")
        if path.suffix == ".py":
            python_count += 1
            try:
                ast.parse(path.read_text(encoding="utf-8-sig"), filename=name)
            except (SyntaxError, UnicodeError) as exc:
                errors.append(f"Invalid Python source {name}: {exc}")
    for required in [
        "README.md", "README.zh-CN.md", "pyproject.toml", ".gitignore",
        "THIRD_PARTY_NOTICES.md", "docs/environment.md", "data/README.md",
        "docs/figures/framework.png", "docs/figures/framework.pdf",
    ]:
        if required not in paths:
            errors.append(f"Required release file missing: {required}")
    print(f"Release selection: {len(paths)} files, {total / 1024**2:.2f} MiB")
    print(f"Python syntax checked: {python_count} files")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("Repository structure and file-size checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

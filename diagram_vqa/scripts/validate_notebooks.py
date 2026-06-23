from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate notebooks for clean-project bootstrap readiness.")
    parser.add_argument("--notebooks-dir", type=Path, default=Path("notebooks"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.notebooks_dir.rglob("*.ipynb"))
    if not paths:
        raise SystemExit(f"No notebooks found under {args.notebooks_dir}")

    failures: list[str] = []
    for path in paths:
        nb = json.loads(path.read_text(encoding="utf-8"))
        cells = nb.get("cells", [])
        if not cells:
            failures.append(f"{path}: no cells")
            continue
        first_source = "".join(cells[0].get("source", []))
        has_bootstrap = "_find_project_root" in first_source and "os.chdir(PROJECT_ROOT)" in first_source
        if not has_bootstrap:
            failures.append(f"{path}: missing clean-project bootstrap in first cell")
        print(f"OK structure: {path}")

    if failures:
        print("\nFailures:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print(f"\nValidated {len(paths)} notebooks.")


if __name__ == "__main__":
    main()

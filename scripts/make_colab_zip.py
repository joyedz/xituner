"""Build the upload bundle for Colab.

Exists instead of `Compress-Archive` for one specific reason: PowerShell writes
zip entry names with backslashes, and Linux `unzip` then creates files literally
named `training\\config.py` rather than a `training/` directory. The import then
fails with a confusing ModuleNotFoundError that has nothing to do with the code.

Python's zipfile always writes forward slashes, so the archive unpacks correctly
on Colab.

Run this again after changing any code -- the notebook uploads a snapshot, not
a live link.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "xituner.zip"

INCLUDE_DIRS = ["training", "scripts", "data"]
SKIP_PARTS = {"__pycache__", ".ipynb_checkpoints"}


def should_include(path: Path) -> bool:
    if not path.is_file():
        return False
    if any(part in SKIP_PARTS for part in path.parts):
        return False
    return path.suffix != ".pyc"


def main() -> None:
    files: list[Path] = []
    for d in INCLUDE_DIRS:
        base = ROOT / d
        if not base.exists():
            raise FileNotFoundError(f"missing directory: {base}")
        files.extend(p for p in base.rglob("*") if should_include(p))

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(files):
            # as_posix() is the whole point: forward slashes, always.
            arcname = p.relative_to(ROOT).as_posix()
            zf.write(p, arcname)

    print(f"wrote {OUT.name} ({OUT.stat().st_size / 1024:.1f} KB)")
    with zipfile.ZipFile(OUT) as zf:
        for name in zf.namelist():
            print(f"  {name}")
        bad = [n for n in zf.namelist() if "\\" in n]
        print(
            "\nOK: all entries use forward slashes"
            if not bad
            else f"\nPROBLEM: backslash entries found: {bad}"
        )


if __name__ == "__main__":
    main()

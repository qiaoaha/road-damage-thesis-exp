from __future__ import annotations

import argparse
import hashlib
import tarfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.archive, "w:gz") as tar:
        tar.add(args.report_dir, arcname=args.report_dir.name)
    digest = sha256_file(args.archive)
    args.archive.with_suffix(args.archive.suffix + ".sha256").write_text(f"{digest}  {args.archive.name}\n", encoding="utf-8")
    print(f"ARCHIVE={args.archive}")
    print(f"SHA256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

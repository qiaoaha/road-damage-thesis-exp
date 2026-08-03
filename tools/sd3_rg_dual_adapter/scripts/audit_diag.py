from __future__ import annotations

import subprocess
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1] / "third_party" / "DIAG"
    sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    print(f"DIAG_UPSTREAM_COMMIT={sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations


def main() -> int:
    print("Release manifest should be generated after pytest/ruff/mypy and git safety checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

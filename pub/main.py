from __future__ import annotations

from .tn import main as _legacy_main


def main(argv: list[str] | None = None) -> None:
    _legacy_main(argv)


if __name__ == "__main__":
    main()

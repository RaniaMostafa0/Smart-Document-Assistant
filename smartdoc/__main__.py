from __future__ import annotations

import sys


def main() -> None:
    print(
        "Use one of:\n"
        "  python -m smartdoc.ingest --data_dir data --index_dir index\n"
        "  python -m smartdoc.chat --index_dir index\n"
    )


if __name__ == "__main__":
    sys.exit(main())


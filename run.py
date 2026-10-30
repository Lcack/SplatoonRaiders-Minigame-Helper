"""Use the current Python environment without requiring an editable install."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from raiders_helper.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

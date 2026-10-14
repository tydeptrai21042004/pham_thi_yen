"""Run the benchmark directly from a source checkout."""
from pathlib import Path
import sys

# Support ``python main.py`` without requiring an editable package install.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from watermarklab.benchmark import main

if __name__ == "__main__":
    main()

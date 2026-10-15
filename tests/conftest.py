"""Make src/ importable for the test suite.

pyproject.toml already sets ``pythonpath = ["src"]`` for pytest; this repeats
it so the tests still import cleanly when pytest is invoked with a different
config or rootdir.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).parent.parent / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

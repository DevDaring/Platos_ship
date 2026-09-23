"""
Phase 3 analysis package.

Analysis modules import from `src` (e.g. `from src.extraction import ...`), so
the Code_Phase_3 root has to be importable however the module is invoked —
`python3 analysis/make_tables.py`, `python3 -m analysis.make_tables`, or a
pytest run from the repository root. Inserting the package parent here makes
all three work without per-module path juggling.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

PROJECT_ROOT = _PROJECT_ROOT

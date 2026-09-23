"""
Phase 3 tests.

Every test here runs offline: no API key, no network, no GPU. They cover the
four defects that produced wrong numbers in the reviewed submission —
process-dependent seeding, the flip-rate denominator, the retention-gap sign,
and unmatched revision prompts — plus the new validators.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

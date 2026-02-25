"""
Compatibility entrypoint.

Your host is configured to run `slp-vehicle-defect-mvp/app.py`.
The real app lives in `vehicle-defect-mvp/app.py`, so we forward execution
without requiring any hosting config changes.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


REAL_APP = Path(__file__).resolve().parents[1] / "vehicle-defect-mvp" / "app.py"
REAL_APP_DIR = REAL_APP.parent

# Ensure imports like `vehicle_defect_mvp.*` resolve the same way they would
# when running the real app directly.
sys.path.insert(0, str(REAL_APP_DIR))

runpy.run_path(str(REAL_APP), run_name="__main__")

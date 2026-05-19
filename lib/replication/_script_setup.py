"""Common sys.path setup for RQ entry-point scripts."""

from __future__ import annotations

import sys
from pathlib import Path


def setup_replication_imports() -> Path:
    """Insert ``lib/`` on ``sys.path`` and return ``diffmodel_esem_replication/`` root."""
    root = Path(__file__).resolve().parents[2]
    lib = root / "lib"
    if str(lib) not in sys.path:
        sys.path.insert(0, str(lib))
    return root

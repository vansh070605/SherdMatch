"""Pottery Fragment Matching (pfm): appearance + fracture-edge geometry."""
from pathlib import Path

# On Windows torch's c10.dll fails to initialise if cv2 is loaded first.
try:
    import torch  # noqa: F401
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "outputs"

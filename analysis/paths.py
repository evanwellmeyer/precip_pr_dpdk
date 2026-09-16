"""Local data and model locations used by the reproduction scripts."""

import os
from pathlib import Path


project = Path(__file__).resolve().parents[1]
research = Path(os.environ.get("PRECIP_RESEARCH_DIR", str(project.parent)))
hadgem = research / "HadGEM"
amip = research / "AMIP"
weights = research / "weights"

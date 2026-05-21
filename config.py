"""
Sanjay System Configuration Hub
Centralizing all hyper-parameters, thresholds, and file-system directory paths.
"""
from pathlib import Path

# 🎯 Accuracy Threshold: How identical must the vectors be? (Range: 0.0 to 1.0)
# 0.50 is an optimized baseline for ArcFace using cosine distance calculations.
SIMILARITY_THRESHOLD: float = 0.50

# ⚡ Performance: Process every N-th frame to save massive computation overhead.
# For instance, 5 means we analyze frame 0, frame 5, frame 10, etc.
FRAME_SAMPLE_RATE: int = 5

# 📁 Storage Paths Management (Using robust pathlib.Path instead of basic strings)
BASE_DIR: Path = Path(__file__).resolve().parent
OUTPUT_DIR: Path = BASE_DIR / "outputs"
MODEL_DIR: Path = BASE_DIR / "models"

# Automatically create the output directories if they don't exist yet
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 📐 Tensor Dimensions: Standard sizing expected by the ArcFace backbone network
FACE_IMAGE_SIZE: tuple[int, int] = (112, 112)
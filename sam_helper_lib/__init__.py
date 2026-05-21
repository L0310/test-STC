import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
SEGMENT_ANYTHING_ROOT = os.path.join(PROJECT_ROOT, "segment-anything-main")
if os.path.isdir(SEGMENT_ANYTHING_ROOT) and SEGMENT_ANYTHING_ROOT not in sys.path:
    sys.path.insert(0, SEGMENT_ANYTHING_ROOT)
A2S_ROOT = os.path.join(PROJECT_ROOT, "A2S-v2-main")
if os.path.isdir(A2S_ROOT) and A2S_ROOT not in sys.path:
    sys.path.insert(0, A2S_ROOT)

from .types import SAMCandidate, SAMResult
from .prompt_points import select_five_prompt_points
from .image_ops import extract_red_mask_from_overlay
from .sam_predictor import SAMHelper
from .train_helper import SAMTrainHelper

__all__ = [
    "SAMCandidate",
    "SAMResult",
    "SAMHelper",
    "SAMTrainHelper",
    "select_five_prompt_points",
    "extract_red_mask_from_overlay",
]

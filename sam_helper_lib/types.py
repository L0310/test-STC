from dataclasses import dataclass
from typing import List, Optional

import numpy as np

@dataclass
class SAMCandidate:
    mask: np.ndarray
    score: float
    point_idx: int
    fg_iou: float = 0.0
    heat_iou: float = 0.0
    bg_iou: float = 0.0
    filter_heat_iou: float = 0.0
    filter_bg_iou: float = 0.0
    logits: Optional[np.ndarray] = None
    mask_orig: Optional[np.ndarray] = None
    prob_orig: Optional[np.ndarray] = None

@dataclass
class SAMResult:
    mask: Optional[np.ndarray]
    points_xy: np.ndarray
    candidates: List[SAMCandidate]
    success: bool

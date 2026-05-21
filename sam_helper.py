from sam_helper_lib import (
    SAMCandidate,
    SAMHelper,
    SAMResult,
    SAMTrainHelper,
    extract_red_mask_from_overlay,
    select_five_prompt_points,
)
from sam_helper_lib.cli import main, parse_args

__all__ = [
    "SAMCandidate",
    "SAMResult",
    "SAMHelper",
    "SAMTrainHelper",
    "extract_red_mask_from_overlay",
    "select_five_prompt_points",
    "parse_args",
    "main",
]


if __name__ == "__main__":
    main()

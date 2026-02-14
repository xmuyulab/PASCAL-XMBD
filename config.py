"""
Configuration for the auto-gating + label propagation project.

This module centralizes:
  1) Global hyperparameters used by the pseudo-gating / probability calibration steps
     (e.g., arcsinh transform, confidence selection, numerical eps).
  2) Dataset registry: mapping dataset identifiers -> (marker prior CSV path, AnnData path).
"""

from dataclasses import dataclass
from typing import Dict


# -----------------------------------------------------------------------------
# Model / pipeline hyperparameters
# -----------------------------------------------------------------------------
@dataclass
class AutoAnnoConfig:
    """
    Parameters controlling pseudo-gating and label propagation.

    Attributes
    ----------
    top_percentage : float
        For each predicted cell type, select the top fraction (0~1) of cells ranked by
        confidence to keep as "confident pseudo-labeled" cells.
        Example: 0.15 means keep top 15% most confident cells per predicted class.

    isArcsinhed : bool
        Whether the input expression matrix in AnnData is already arcsinh-transformed.

    cofactor : int
        Cofactor used in arcsinh transform. Common values: 5 (CyTOF/IMC).

    eps : float
        Small constant for numerical stability (avoid divide-by-zero / log(0)).

    results_path : str
        Output directory for saving results.

    unknown_thres : int
        Minimum confidence ratio to accept a predicted label.
        Confidence is defined as:
            conf = p1 / p2
        If conf < unknown_thres, the prediction is considered ambiguous and the cell
        should be treated as Unknown.

        Default: 2.0

    seed : int
        Global random seed used for reproducibility (e.g., GMM initialization).
    """
    top_percentage: float
    isArcsinhed: bool = True
    cofactor: int = 5
    eps: float = 1e-12
    results_path: str = "./results"
    unknown_thres: int = 2
    seed: int = 0


params = AutoAnnoConfig(
    top_percentage=0.15,     # keep top 15% most confident cells per predicted class
    isArcsinhed=True,        # input matrix already arcsinh-transformed
    cofactor=5,              # only used when isArcsinhed=False
    eps=1e-12,               # numerical stability constant
    results_path="./results",
    unknown_thres=2,
    seed=0
)


# -----------------------------------------------------------------------------
# Dataset registries
# -----------------------------------------------------------------------------
# MARKER_PRIOR_PATHS:
#   dataset_id -> marker prior CSV path
MARKER_PRIOR_PATHS: Dict[str, str] = {
    "IMC": "gating_strategy/IMC_prior.csv",
}

# TEST_DATA_PATHS:
#   dataset_id -> AnnData .h5ad file path
TEST_DATA_PATHS: Dict[str, str] = {
    "IMC": "example/IMC_data.h5ad",
}

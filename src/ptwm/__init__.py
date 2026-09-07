"""ptwm: reduced-state and tensor-network world models, Experiment A."""

__version__ = "0.1.0"

from .data import Episode, EpisodeDataset, load_cell, load_full_json
from .metrics import (
    fidelity_mae,
    horizon_of_tolerance,
    log_mse,
    per_length_errors,
    physicality_residuals,
    summarize_all,
)
from .models import (
    BaseModel,
    GRUModel,
    MarkovChannel,
    ProcessMPO,
    TransferTensor,
    TransformerModel,
)
from .splits import all_splits, bias_split, family_split, horizon_split

__all__ = [
    "Episode",
    "EpisodeDataset",
    "load_cell",
    "load_full_json",
    "log_mse",
    "fidelity_mae",
    "per_length_errors",
    "horizon_of_tolerance",
    "physicality_residuals",
    "summarize_all",
    "BaseModel",
    "MarkovChannel",
    "TransferTensor",
    "ProcessMPO",
    "GRUModel",
    "TransformerModel",
    "all_splits",
    "horizon_split",
    "bias_split",
    "family_split",
]

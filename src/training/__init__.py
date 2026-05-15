"""Phase 2 training utilities: BPR loss + GroupAggregatorTrainer."""

from .bpr_loss import pairwise_bpr_loss
from .group_trainer import (
    GroupAggregatorTrainer,
    GroupTrainConfig,
    GroupTrainDataset,
    GroupEvalDataset,
    build_user_score_lookup,
    compute_pop_counts,
    lookup_per_user_scores,
)

__all__ = [
    "pairwise_bpr_loss",
    "GroupAggregatorTrainer",
    "GroupTrainConfig",
    "GroupTrainDataset",
    "GroupEvalDataset",
    "build_user_score_lookup",
    "compute_pop_counts",
    "lookup_per_user_scores",
]

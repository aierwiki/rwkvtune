"""
GRPO (Group Relative Policy Optimization) auxiliary modules.

This module contains GRPO-specific auxiliary components:
- lightning_module: GRPO Lightning module
- dataset: Data loading utilities
- loss: Loss functions
- advantage: Advantage calculation

Note: GRPOConfig and GRPOTrainer are at the rwkvtune.training level.

Usage:
    from rwkvtune.training import GRPOConfig, GRPOTrainer
"""
from rwkvtune.training.grpo.lightning_module import GRPOLightningModule

__all__ = [
    "GRPOLightningModule",
]

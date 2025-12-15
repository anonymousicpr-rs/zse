"""
ZAYAN: Disentangled Contrastive Transformer for Tabular Remote Sensing Data

Public API:
- ZAYAN_CL : Feature-level contrastive learning module.
- ZAYAN_T  : Transformer-based classifier module.
- ZAYAN    : High-level wrapper that runs CL pretraining and supervised training.
"""

from .zayan import ZAYAN_CL, ZAYAN_T, ZAYAN

__all__ = ["ZAYAN_CL", "ZAYAN_T", "ZAYAN"]

__version__ = "0.1.0"
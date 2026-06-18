import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

def _check_finite(name, x):
    """
        Simple check to find Inf/NaN Values in tensors
    """
    if not torch.isfinite(x).all():
        logger.debug(f"NaN detected in {name}")

def check_finite(name: str, x: Any, enabled: bool=True) -> None:
    """
        Recursively check for NaN/Inf values in an iterable containing tensors
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
    if isinstance(x, dict):
        for k, v in x.items():
            check_finite(f"{name}.{k}", v)
    elif isinstance(x, list):
        for idx, v in enumerate(x):
            check_finite(f"{name}.{idx}", v)
    elif isinstance(x, torch.Tensor):
        _check_finite(name, x)
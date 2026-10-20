from collections.abc import Mapping
from typing import Tuple, List, Dict, Any

import torch

def to_device(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device)

    if isinstance(obj, Mapping):
        return {k: to_device(v, device) for k, v in obj.items()}

    if isinstance(obj, list):
        return [to_device(v, device) for v in obj]

    if isinstance(obj, tuple):
        return tuple(to_device(v, device) for v in obj)

    return obj
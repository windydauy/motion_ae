"""随机种子设置。"""
from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """固定所有随机源。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

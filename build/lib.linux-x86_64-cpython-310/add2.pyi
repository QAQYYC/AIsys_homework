import torch
from typing import Any

def torch_launch_add2(
        c: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        n: int
) -> None: ...
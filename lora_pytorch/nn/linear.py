from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from lora_pytorch.nn.base import BaseLoRAModule


class LinearLoRAModule(BaseLoRAModule[nn.Linear]):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.rank = rank
        self.in_proj = nn.Parameter(
            torch.empty((in_features, rank), device=device, dtype=dtype),
            requires_grad=True,
        )
        self.out_proj = nn.Parameter(
            torch.empty((rank, out_features), device=device, dtype=dtype),
            requires_grad=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.in_proj @ self.out_proj

    @torch.no_grad()
    def merge(self, module: nn.Linear, inplace: bool = False) -> nn.Linear:
        weight = module.weight.data + self.in_proj.data @ self.out_proj.data
        if inplace:
            module.weight.data = weight
            return module

        out = nn.Linear(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        out.weight.data = weight
        out.bias.data = module.bias.data
        return out

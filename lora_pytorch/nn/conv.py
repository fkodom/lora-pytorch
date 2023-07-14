from __future__ import annotations

from copy import deepcopy
from typing import Optional, TypeVar, Union

import torch
from einops import einsum
from torch import Tensor, nn

from lora_pytorch.nn.base import BaseLoRAModule

Conv = Union[nn.Conv1d, nn.Conv2d, nn.Conv3d]
ConvType = TypeVar("ConvType", nn.Conv1d, nn.Conv2d, nn.Conv3d)


class ConvLoRAModule(BaseLoRAModule[Conv]):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rank: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.rank = rank
        self.in_proj = nn.Parameter(
            torch.empty((in_channels, rank), device=device, dtype=dtype),
            requires_grad=True,
        )
        self.out_proj = nn.Parameter(
            torch.empty((rank, out_channels), device=device, dtype=dtype),
            requires_grad=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        # einstein notation:
        # - b: batch size
        # - c: channels
        # - r: rank
        x = einsum(x, self.in_proj, "b c ..., c r -> b r ...")
        x = einsum(x, self.out_proj, "b r ..., r c -> b c ...")
        return x @ self.in_proj @ self.out_proj

    @torch.no_grad()
    def merge(self, module: ConvType, inplace: bool = False) -> ConvType:  # type: ignore
        # einstein notation:
        # - i: in channels
        # - o: out channels
        # - r: rank
        lora_weight = einsum(self.in_proj.data, self.out_proj.data, "i r, r o -> o i")
        # NOTE: We need to normalize the weight, since the convolution kernel
        # will be applied over the spatial dimensions.  Get the spatial extent
        # of the kernel, and divide the lora weight by that.
        kernel_size = torch.tensor(module.weight.shape[2:]).prod().item()
        lora_weight = lora_weight / kernel_size
        # Add lora weights to the existing kernel weights.
        weight = einsum(module.weight.data, lora_weight, "o i ..., o i -> o i ...")

        if inplace:
            module.weight.data = weight
            return module

        out_module = deepcopy(module)
        out_module.weight.data = weight
        return out_module

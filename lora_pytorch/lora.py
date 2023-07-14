from __future__ import annotations

from typing import Generic, Optional, TypeVar

import torch
from torch import Tensor, nn

from lora_pytorch.nn.base import BaseLoRAModule
from lora_pytorch.nn.conv import ConvLoRAModule, ConvType
from lora_pytorch.nn.linear import LinearLoRAModule

ModuleType = TypeVar("ModuleType", bound=nn.Module)


class LoRA(nn.Module, Generic[ModuleType]):
    def __init__(
        self,
        module: ModuleType,
        lora_module: Optional[BaseLoRAModule],
        enabled: bool = True,
    ):
        super().__init__()
        self.module = module.eval()
        self.lora_module = lora_module
        self.enabled = enabled

    def forward(self, x: Tensor) -> Tensor:
        enable_grad = (not self.enabled) and torch.is_grad_enabled()
        with torch.set_grad_enabled(enable_grad):
            x = self.module(x)
        if self.enabled and self.lora_module is not None:
            x = x + self.lora_module(x)
        return x

    def enable_lora(self):
        if self.lora_module is None:
            raise ValueError("Cannot enable LoRA when self.lora_module is None")
        self.enabled = True

    def disable_lora(self):
        self.enabled = False

    def remove_lora(self) -> ModuleType:
        return self.module

    def merge_lora(self, inplace: bool = False) -> ModuleType:
        if self.lora_module is None:
            return self.module
        else:
            return self.lora_module.merge(self.module, inplace=inplace)

    @classmethod
    def _from_linear(cls, module: nn.Linear, rank: int) -> LoRA[nn.Linear]:
        out_size, in_size = module.weight.shape
        device = module.weight.device
        dtype = module.weight.dtype
        lora_module = LinearLoRAModule(
            in_size, out_size, rank=rank, device=device, dtype=dtype
        )
        return LoRA(module, lora_module)

    @classmethod
    def _from_conv(cls, module: ConvType, rank: int) -> LoRA[ConvType]:
        out_channels, in_channels = module.weight.shape[:2]
        device = module.weight.device
        dtype = module.weight.dtype
        lora_module = ConvLoRAModule(
            in_channels=in_channels,
            out_channels=out_channels,
            rank=rank,
            device=device,
            dtype=dtype,
        )
        return LoRA(module, lora_module)

    @classmethod
    def from_module(cls, module: ModuleType, rank: int) -> LoRA[ModuleType]:
        if isinstance(module, nn.Linear):
            return LoRA._from_linear(module, rank)
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            return LoRA._from_conv(module, rank)
        elif isinstance(module, nn.Module):
            return LoRA(module, None, enabled=False)
        else:
            raise ValueError(f"Unsupported module type: {type(module)}")


if __name__ == "__main__":
    linear = nn.Linear(10, 10)
    lora = LoRA._from_linear(linear, rank=5)

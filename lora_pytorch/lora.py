from __future__ import annotations

from typing import Generic, Literal, Optional, Type, TypeVar, Union, cast, overload

import torch
from torch import Tensor, nn

from lora_pytorch.modules.base import BaseLoRAModule
from lora_pytorch.modules.conv import Conv1dLoRA, Conv2dLoRA, Conv3dLoRA, ConvType
from lora_pytorch.modules.linear import LinearLoRAModule

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
            y = self.module(x)
        if self.enabled and self.lora_module is not None:
            y = y + self.lora_module(x)
        return y

    def enable_lora(self) -> None:
        if self.lora_module is None:
            raise ValueError("Cannot enable LoRA when self.lora_module is None")

        for _, child in self.module.named_children():
            if isinstance(child, LoRA) and child.lora_module is not None:
                child.enable_lora()

        self.enabled = True

    def disable_lora(self) -> None:
        for _, child in self.module.named_children():
            if isinstance(child, LoRA):
                child.disable_lora()

        self.enabled = False

    # TODO: Add option for 'inplace=False'
    def remove_lora(self) -> ModuleType:
        """Remove all LoRA modules from the model.

        NOTE: This is an in-place operation!  This is not easily reversible, and
        it will affect all references to the model, or its child layers.
        """
        for name, child in self.module.named_children():
            if isinstance(child, LoRA):
                self.module._modules[name] = child.remove_lora()

        return self.module

    def merge_lora(self, inplace: bool = False) -> ModuleType:
        for name, child in self.module.named_children():
            if isinstance(child, LoRA):
                self.module._modules[name] = child.merge_lora(inplace=inplace)

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
        out_channels, in_channels, *_ = module.weight.shape
        device = module.weight.device
        dtype = module.weight.dtype

        lora_module_cls: Union[Type[Conv1dLoRA], Type[Conv2dLoRA], Type[Conv3dLoRA]]
        if isinstance(module, nn.Conv1d):
            lora_module_cls = Conv1dLoRA
        elif isinstance(module, nn.Conv2d):
            lora_module_cls = Conv2dLoRA
        elif isinstance(module, nn.Conv3d):
            lora_module_cls = Conv3dLoRA
        else:
            raise ValueError(f"Unsupported conv module type: {type(module)}")

        lora_module = lora_module_cls(
            in_channels=in_channels,
            out_channels=out_channels,
            rank=rank,
            kernel_size=module.kernel_size,  # type: ignore
            stride=module.stride,  # type: ignore
            padding=module.padding,  # type: ignore
            dilation=module.dilation,  # type: ignore
            # TODO: Add support for groups
            # groups=module.groups,
            device=device,
            dtype=dtype,
        )

        return LoRA(module, lora_module)

    @overload
    @classmethod
    def from_module(
        cls, module: ModuleType, rank: int, is_root: Literal[True] = True
    ) -> LoRA[ModuleType]:
        ...

    @overload
    @classmethod
    def from_module(
        cls, module: ModuleType, rank: int, is_root: Literal[False] = False
    ) -> Union[LoRA[ModuleType], ModuleType]:
        ...

    @classmethod
    def from_module(cls, module: ModuleType, rank: int, is_root: bool = True):
        if isinstance(module, nn.Linear):
            return LoRA._from_linear(module, rank)  # type: ignore
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            return LoRA._from_conv(module, rank)  # type: ignore

        for name, child in module.named_children():
            child = cast(ModuleType, child)
            setattr(module, name, cls.from_module(child, rank, is_root=False))

        is_lora = any(isinstance(child, LoRA) for child in module.children())
        if is_lora or is_root:
            return LoRA(module, None, enabled=False)
        else:
            return module


if __name__ == "__main__":
    from torchvision.models import resnet18

    model = resnet18().cuda()
    model.eval()

    lora = LoRA.from_module(model, rank=1)
    x = torch.randn(1, 3, 224, 224).cuda()
    y = lora(x)
    print(lora)

    breakpoint()
    pass

from __future__ import annotations

from copy import deepcopy
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
        self.enabled = enabled and lora_module is not None

        if not enabled:
            self.disable_lora()
        else:
            self.enable_lora()

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        if self.enabled and args:
            raise ValueError("LoRA modules do not support positional arguments.")

        enable_grad = (not self.enabled) and torch.is_grad_enabled()
        with torch.set_grad_enabled(enable_grad):
            y = self.module(x, *args, **kwargs)
        if self.enabled and self.lora_module is not None:
            y = y + self.lora_module(x)

        return y

    def enable_lora(self) -> None:
        return enable_lora(self)  # type: ignore

    def disable_lora(self) -> None:
        return disable_lora(self)  # type: ignore

    def remove_lora(self, inplace: bool = False) -> ModuleType:
        """Remove all LoRA modules from the model."""
        return remove_lora(self, inplace=inplace)  # type: ignore

    def merge_lora(self: LoRA[ModuleType], inplace: bool = False) -> ModuleType:
        return merge_lora(self, inplace=inplace)  # type: ignore

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
            in_channels=module.in_channels,
            out_channels=module.out_channels,
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
        cls,
        module: ModuleType,
        rank: int,
        enabled: bool = True,
        is_root: Literal[True] = True,
    ) -> LoRA[ModuleType]:
        ...

    @overload
    @classmethod
    def from_module(
        cls,
        module: ModuleType,
        rank: int,
        enabled: bool = True,
        is_root: Literal[False] = False,
    ) -> Union[LoRA[ModuleType], ModuleType]:
        ...

    @classmethod
    def from_module(
        cls, module: ModuleType, rank: int, enabled: bool = True, is_root: bool = True
    ):
        if isinstance(module, nn.Linear):
            return LoRA._from_linear(module, rank)  # type: ignore
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            return LoRA._from_conv(module, rank)  # type: ignore

        for name, child in module.named_children():
            child = cast(ModuleType, child)
            module._modules[name] = cls.from_module(
                child, rank, enabled=enabled, is_root=False
            )

        if is_root:
            return LoRA(module, None, enabled=enabled)
        else:
            return module


def enable_lora(module: Union[ModuleType, LoRA[ModuleType]]) -> None:
    for child in module.children():
        enable_lora(child)

    if isinstance(module, LoRA):
        module.enabled = True


def disable_lora(module: Union[ModuleType, LoRA[ModuleType]]) -> None:
    for child in module.children():
        disable_lora(child)

    if isinstance(module, LoRA):
        module.enabled = False


def merge_lora(
    module: Union[ModuleType, LoRA[ModuleType]], inplace: bool = False
) -> ModuleType:
    if not inplace:
        module = deepcopy(module)

    for name, child in module.named_children():
        module._modules[name] = merge_lora(child)

    if isinstance(module, LoRA):
        if module.lora_module is None:
            return module.module
        else:
            return module.lora_module.merge(module.module, inplace=True)
    else:
        return module


def remove_lora(
    module: Union[ModuleType, LoRA[ModuleType]], inplace: bool = False
) -> ModuleType:
    """Remove all LoRA modules from the model."""
    if not inplace:
        module = deepcopy(module)

    if isinstance(module, LoRA):
        return remove_lora(module.module, inplace=True)

    for name, child in module.named_children():
        module._modules[name] = remove_lora(child, inplace=True)

    return module

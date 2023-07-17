from copy import deepcopy
from typing import Generator, Tuple

import pytest
import torch
from torch import nn

from lora_pytorch.modules.conv import (
    Conv1dLoRAModule,
    Conv2dLoRAModule,
    Conv3dLoRAModule,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module", params=[1, 3, 8])
def in_channels(request) -> Generator[int, None, None]:
    yield request.param


@pytest.fixture(scope="module", params=[1, 3, 8])
def out_channels(request) -> Generator[int, None, None]:
    yield request.param


@pytest.fixture(scope="module", params=[1, 3])
def rank(request) -> Generator[int, None, None]:
    yield request.param


@pytest.fixture(
    scope="module",
    params=[
        # (kernel_size, stride, padding, dilation, bias)
        (1, 1, 0, 1, False),
        (1, 1, 0, 1, True),
        (3, 1, 0, 1, True),
        (3, 1, 1, 1, True),
        (3, 2, 1, 1, True),
    ],
)
def conv1d_and_lora_module(
    request, in_channels: int, out_channels: int, rank: int
) -> Generator[Tuple[nn.Conv1d, Conv1dLoRAModule], None, None]:
    kernel_size, stride, padding, dilation, bias = request.param
    conv = nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        device=DEVICE,
    ).eval()
    lora_module = Conv1dLoRAModule(
        in_channels=in_channels,
        out_channels=out_channels,
        rank=rank,
        kernel_size=conv.kernel_size,  # type: ignore
        stride=conv.stride,  # type: ignore
        padding=conv.padding,  # type: ignore
        dilation=conv.dilation,  # type: ignore
        device=DEVICE,
    ).eval()
    yield conv, lora_module


@torch.no_grad()
def test_conv1d_merge(conv1d_and_lora_module: Tuple[nn.Conv1d, Conv1dLoRAModule]):
    conv, lora_module = conv1d_and_lora_module
    x = torch.randn(1, conv.in_channels, 5, device=DEVICE)
    y1 = conv(x) + lora_module(x)

    merged = lora_module.merge(conv, inplace=False)
    y2 = merged(x)
    assert torch.allclose(y1, y2, atol=1e-5)

    conv_copy = deepcopy(conv)
    merged = lora_module.merge(conv_copy, inplace=True)
    y3 = merged(x)
    assert torch.allclose(y1, y3)


@pytest.fixture(
    scope="module",
    params=[
        # (kernel_size, stride, padding, dilation, bias)
        (1, 1, 0, 1, False),
        (1, 1, 0, 1, True),
        (3, 1, 1, 1, True),
        (3, 2, 1, 1, True),
        ((1, 3), 1, 0, 1, True),
    ],
)
def conv2d_and_lora_module(
    request, in_channels: int, out_channels: int, rank: int
) -> Generator[Tuple[nn.Conv2d, Conv2dLoRAModule], None, None]:
    kernel_size, stride, padding, dilation, bias = request.param
    conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        device=DEVICE,
    ).eval()
    lora_module = Conv2dLoRAModule(
        in_channels=in_channels,
        out_channels=out_channels,
        rank=rank,
        kernel_size=conv.kernel_size,  # type: ignore
        stride=conv.stride,  # type: ignore
        padding=conv.padding,  # type: ignore
        dilation=conv.dilation,  # type: ignore
        device=DEVICE,
    ).eval()
    yield conv, lora_module


@torch.no_grad()
def test_conv2d_merge(conv2d_and_lora_module: Tuple[nn.Conv2d, Conv2dLoRAModule]):
    conv, lora_module = conv2d_and_lora_module
    x = torch.randn(1, conv.in_channels, 5, 5, device=DEVICE)
    y1 = conv(x) + lora_module(x)

    merged = lora_module.merge(conv, inplace=False)
    y2 = merged(x)
    assert torch.allclose(y1, y2, atol=1e-5)

    conv_copy = deepcopy(conv)
    merged = lora_module.merge(conv_copy, inplace=True)
    y3 = merged(x)
    assert torch.allclose(y1, y3)


@pytest.fixture(
    scope="module",
    params=[
        (1, 1, 0, 1, False),
        (1, 1, 0, 1, True),
        (3, 1, 1, 1, True),
        (3, 2, 1, 1, True),
        ((1, 3, 3), 1, 0, 1, True),
    ],
)
def conv3d_and_lora_module(
    request, in_channels: int, out_channels: int, rank: int
) -> Generator[Tuple[nn.Conv3d, Conv3dLoRAModule], None, None]:
    kernel_size, stride, padding, dilation, bias = request.param
    conv = nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        device=DEVICE,
    ).eval()
    lora_module = Conv3dLoRAModule(
        in_channels=in_channels,
        out_channels=out_channels,
        rank=rank,
        kernel_size=conv.kernel_size,  # type: ignore
        stride=conv.stride,  # type: ignore
        padding=conv.padding,  # type: ignore
        dilation=conv.dilation,  # type: ignore
        device=DEVICE,
    ).eval()
    yield conv, lora_module


@torch.no_grad()
def test_conv3d_merge(conv3d_and_lora_module: Tuple[nn.Conv3d, Conv3dLoRAModule]):
    conv, lora_module = conv3d_and_lora_module
    x = torch.randn(1, conv.in_channels, 5, 5, 5, device=DEVICE)
    y1 = conv(x) + lora_module(x)

    merged = lora_module.merge(conv, inplace=False)
    y2 = merged(x)
    assert torch.allclose(y1, y2, atol=1e-5)

    conv_copy = deepcopy(conv)
    merged = lora_module.merge(conv_copy, inplace=True)
    y3 = merged(x)
    assert torch.allclose(y1, y3)

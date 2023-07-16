from copy import deepcopy
from typing import Generator, Tuple

import pytest
import torch
from torch import nn

from lora_pytorch.modules.linear import LinearLoRAModule

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module", params=[1, 3, 8])
def in_features(request) -> Generator[int, None, None]:
    yield request.param


@pytest.fixture(scope="module", params=[1, 4, 7])
def out_features(request) -> Generator[int, None, None]:
    yield request.param


@pytest.fixture(scope="module", params=[1, 2, 3])
def rank(request) -> Generator[int, None, None]:
    yield request.param


@pytest.fixture(scope="module", params=[True, False])
def bias(request) -> Generator[bool, None, None]:
    yield request.param


@pytest.fixture(scope="module")
def linear_and_lora_module(
    in_features: int, out_features: int, rank: int, bias: bool
) -> Generator[Tuple[nn.Linear, LinearLoRAModule], None, None]:
    linear = nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        device=DEVICE,
    ).eval()
    lora_module = LinearLoRAModule(
        in_features=in_features,
        out_features=out_features,
        rank=rank,
        device=DEVICE,
    ).eval()
    yield linear, lora_module


@torch.no_grad()
def test_merge(linear_and_lora_module: Tuple[nn.Linear, LinearLoRAModule]):
    linear, lora_module = linear_and_lora_module
    in_features = linear.in_features
    x = torch.randn(1, in_features, device=DEVICE)
    y1 = linear(x) + lora_module(x)

    merged = lora_module.merge(linear, inplace=False)
    y2 = merged(x)
    assert torch.allclose(y1, y2)

    linear_copy = deepcopy(linear)
    merged = lora_module.merge(linear_copy, inplace=True)
    y3 = merged(x)
    assert torch.allclose(y1, y3)


# TODO
# def test_grad

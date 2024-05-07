from copy import deepcopy
from functools import partial
from typing import Callable, Generator, Tuple

import pytest
import torch
from pytest import FixtureRequest
from torch import Tensor, nn
from torchtext.functional import to_tensor
from torchtext.models import XLMR_BASE_ENCODER
from torchvision.models import ResNet18_Weights, ViT_B_32_Weights, resnet18, vit_b_32

from lora_pytorch.lora import LoRA

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64  # for better precision/stability when testing LoRA
# Force CUDA ops to be deterministic
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def _init_random_lora_module(module: nn.Module):
    for child in module.children():
        _init_random_lora_module(child)

    if isinstance(module, LoRA) and module.lora_module is not None:
        for param in module.lora_module.parameters():
            param.data = nn.init.uniform_(param, a=-0.1, b=0.1)


@pytest.fixture(params=[1, 2, 4])
def rank(request) -> Generator[int, None, None]:
    yield request.param


@torch.no_grad()
@pytest.mark.parametrize("embed_dim", [8, 16])
@pytest.mark.parametrize("num_heads", [1, 2])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("kvdim", [8, 16])
@pytest.mark.parametrize("batch_first", [True, False])
def test_multihead_attention(
    rank: int,
    embed_dim: int,
    num_heads: int,
    bias: bool,
    kvdim: int,
    batch_first: bool,
):
    model = nn.MultiheadAttention(
        embed_dim,
        num_heads,
        bias=bias,
        kdim=kvdim,
        vdim=kvdim,
        batch_first=batch_first,
        device=DEVICE,
        dtype=DTYPE,
    ).eval()
    if batch_first:
        q = torch.randn(1, 16, embed_dim, device=DEVICE, dtype=DTYPE)
        kv = torch.randn(1, 16, kvdim, device=DEVICE, dtype=DTYPE)
    else:
        q = torch.randn(16, 1, embed_dim, device=DEVICE, dtype=DTYPE)
        kv = torch.randn(16, 1, kvdim, device=DEVICE, dtype=DTYPE)

    y1, _ = model(q, kv, kv)

    lora = LoRA.from_module(model, rank=rank)
    lora.eval()
    y2, _ = lora(q, kv, kv)
    # By default, LoRA is initialized so that output is the same as the original model.
    assert torch.allclose(y1, y2)
    # In order to test enable/disable/remove/merge functions, we need to randomly
    # initialize the LoRA weights, so that the outputs are different.
    _init_random_lora_module(lora)
    y3, _ = lora(q, kv, kv)
    assert not torch.allclose(y1, y3, atol=1e-5)

    lora.disable_lora()
    y4, _ = lora(q, kv, kv)
    assert torch.allclose(y1, y4, atol=1e-5)

    lora.enable_lora()
    y5, _ = lora(q, kv, kv)
    assert torch.allclose(y3, y5, atol=1e-4)

    merged = lora.merge_lora(inplace=False)
    assert not isinstance(merged, LoRA)
    y6, _ = merged(q, kv, kv)
    assert torch.allclose(y5, y6, atol=1e-4)

    lora_copy = deepcopy(lora)
    merged2 = lora_copy.merge_lora(inplace=True)
    assert not isinstance(merged2, LoRA)
    y7, _ = merged2(q, kv, kv)
    assert torch.allclose(y5, y7, atol=1e-4)

    removed = lora.remove_lora()
    y8, _ = removed(q, kv, kv)
    assert torch.allclose(y1, y8, atol=1e-4)


@pytest.fixture(
    params=[
        partial(resnet18, weights=ResNet18_Weights.DEFAULT),
        partial(vit_b_32, weights=ViT_B_32_Weights.DEFAULT),
        # TODO: If we figure out how to use 'rank < groups' in LoRA, we can also
        # test models with grouped convolutions. e.g.:
        # partial(mobilenet_v3_small, weights=MobileNet_V3_Small_Weights.DEFAULT),
    ],
)
def vision_model(request: FixtureRequest) -> Generator[nn.Module, None, None]:
    yield request.param().eval().to(device=DEVICE, dtype=DTYPE)


@torch.no_grad()
def test_vision_model(vision_model: nn.Module, rank: int):
    x = torch.randn(1, 3, 224, 224, device=DEVICE, dtype=DTYPE)
    y1 = vision_model(x)

    lora = LoRA.from_module(vision_model, rank=rank)
    lora.eval()
    y2 = lora(x)
    # By default, LoRA is initialized so that output is the same as the original model.
    assert torch.allclose(y1, y2)
    # In order to test enable/disable/remove/merge functions, we need to randomly
    # initialize the LoRA weights, so that the outputs are different.
    _init_random_lora_module(lora)
    y3 = lora(x)
    assert not torch.allclose(y1, y3, atol=1e-5)

    lora.disable_lora()
    y4 = lora(x)
    assert torch.allclose(y1, y4, atol=1e-5)

    lora.enable_lora()
    y5 = lora(x)
    torch.testing.assert_allclose(y3, y5, rtol=1e-4, atol=1e-4)

    merged = lora.merge_lora(inplace=False)
    assert not isinstance(merged, LoRA)
    y6 = merged(x)
    torch.testing.assert_allclose(y5, y6, rtol=1e-4, atol=1e-4)

    lora_copy = deepcopy(lora)
    merged2 = lora_copy.merge_lora(inplace=True)
    assert not isinstance(merged2, LoRA)
    y7 = merged2(x)
    assert torch.allclose(y5, y7, atol=1e-4)

    removed = lora.remove_lora()
    y8 = removed(x)
    assert torch.allclose(y1, y8, atol=1e-4)


@pytest.fixture(
    params=[XLMR_BASE_ENCODER],
)
def text_model(
    request,
) -> Generator[Tuple[nn.Module, Callable[[str], Tensor]], None, None]:
    bundle = request.param
    model = bundle.get_model().eval().to(DEVICE)
    transform = bundle.transform()
    yield model, transform


@torch.no_grad()
def test_text_model(text_model: Tuple[nn.Module, Callable[[str], Tensor]], rank: int):
    model, transform = text_model
    texts = ["Hello, world!"]
    x = to_tensor(transform(texts)).to(DEVICE)

    y1 = model(x)

    lora = LoRA.from_module(model, rank=rank)
    lora.eval()
    y2 = lora(x)
    # By default, LoRA is initialized so that output is the same as the original model
    assert torch.allclose(y1, y2)
    # In order to test enable/disable/remove/merge functions, we need to randomly
    # initialize the LoRA weights, so that the outputs are different.
    _init_random_lora_module(lora)
    y3 = lora(x)
    assert not torch.allclose(y1, y3, atol=1e-5)

    lora.disable_lora()
    y4 = lora(x)
    assert torch.allclose(y1, y4, atol=1e-5)

    lora.enable_lora()
    y5 = lora(x)
    assert torch.allclose(y3, y5, atol=1e-4)

    merged = lora.merge_lora(inplace=False)
    assert not isinstance(merged, LoRA)
    y6 = merged(x)
    assert torch.allclose(y5, y6, atol=1e-4)

    lora_copy = deepcopy(lora)
    merged2 = lora_copy.merge_lora(inplace=True)
    assert not isinstance(merged2, LoRA)
    y7 = merged2(x)
    assert torch.allclose(y5, y7, atol=1e-4)

    removed = lora.remove_lora()
    y8 = removed(x)
    assert torch.allclose(y1, y8, atol=1e-4)

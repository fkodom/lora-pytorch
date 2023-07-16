from copy import deepcopy
from functools import partial
from typing import Callable, Generator, Tuple

import pytest
import torch
from torch import Tensor, nn
from torchtext.models import ROBERTA_BASE_ENCODER, RobertaModel
from torchvision.models import resnet18

from lora_pytorch.lora import LoRA

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Force CUDA ops to be deterministic
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


@pytest.fixture(scope="module", params=[1, 2, 3])
def rank(request) -> Generator[int, None, None]:
    yield request.param


@pytest.fixture(
    scope="module",
    params=[
        partial(resnet18, pretrained=True),
        # TODO: After adding support for 'groups' in conv operators
        # partial(mobilenet_v3_small, pretrained=True),
        # partial(vit_b_32, pretrained=False),
    ],
)
def vision_model(request) -> Generator[nn.Module, None, None]:
    yield request.param().eval().to(DEVICE)


@pytest.fixture(
    scope="module",
    params=[ROBERTA_BASE_ENCODER],
)
def text_model(
    request,
) -> Generator[Tuple[RobertaModel, Callable[[str], Tensor]], None, None]:
    model = request.param.get_model().to(DEVICE)
    transform = request.param.transform()
    yield model, transform


def init_random_lora_module(lora: LoRA):
    for child in lora.module.children():
        if isinstance(child, LoRA):
            init_random_lora_module(child)

    if lora.lora_module is not None:
        for param in lora.lora_module.parameters():
            param.data = nn.init.uniform_(param, a=-0.1, b=0.1)


@torch.no_grad()
def test_vision_model(vision_model: nn.Module, rank: int):
    x = torch.randn(1, 3, 224, 224, device=DEVICE)
    y1 = vision_model(x)

    lora = LoRA.from_module(vision_model, rank=rank)
    lora.eval()
    y2 = lora(x)
    # By default, LoRA is initialized so that output is the same as the original model.
    assert torch.allclose(y1, y2)
    # In order to test enable/disable/remove/merge functions, we need to randomly
    # initialize the LoRA weights, so that the outputs are different.
    init_random_lora_module(lora)
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

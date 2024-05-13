from copy import deepcopy
from typing import Generator, Tuple

import pytest
import torch
from torch import nn

from lora_pytorch.modules.embedding import EmbeddingLoRAModule

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module", params=[8, 32, 128])
def num_embeddings(request) -> Generator[int, None, None]:
    yield request.param


@pytest.fixture(scope="module", params=[4, 16, 64])
def embedding_dim(request) -> Generator[int, None, None]:
    yield request.param


@pytest.fixture(scope="module", params=[1, 2, 3])
def rank(request) -> Generator[int, None, None]:
    yield request.param


@pytest.fixture(scope="module", params=[True, False])
def bias(request) -> Generator[bool, None, None]:
    yield request.param


@pytest.fixture(scope="module")
def embedding_and_lora_module(
    num_embeddings: int, embedding_dim: int, rank: int, bias: bool
) -> Generator[Tuple[nn.Embedding, EmbeddingLoRAModule], None, None]:
    linear = nn.Embedding(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        device=DEVICE,
    ).eval()
    lora_module = EmbeddingLoRAModule(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        rank=rank,
        device=DEVICE,
    ).eval()
    yield linear, lora_module


@torch.no_grad()
def test_merge(embedding_and_lora_module: Tuple[nn.Embedding, EmbeddingLoRAModule]):
    embedding, lora_module = embedding_and_lora_module
    num_embeddings = embedding.num_embeddings
    x = torch.randint(0, num_embeddings - 1, size=(1, 1024), device=DEVICE)
    y1 = embedding(x) + lora_module(x)

    merged = lora_module.merge(embedding, inplace=False)
    y2 = merged(x)
    assert torch.allclose(y1, y2)

    embedding_copy = deepcopy(embedding)
    merged = lora_module.merge(embedding_copy, inplace=True)
    y3 = merged(x)
    assert torch.allclose(y1, y3)


# TODO
# def test_grad

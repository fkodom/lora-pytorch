# lora-pytorch

A simple, robust implementation of [LoRA (Low-Rank Adaptation)](https://arxiv.org/pdf/2106.09685.pdf) for PyTorch.
* Compatible with NLP, CV, and other model types ✔️
* Strongly typed ✔️
* Fully tested ✔️


## Install

PyPI:
```bash
pip install lora-pytorch
```

From source:
```bash
pip install "lora-pytorch @ git+ssh://git@github.com/fkodom/lora-pytorch.git"
```

For contributors:
```bash
# Clone repository
gh repo clone fkodom/lora-pytorch
# Install all dev dependencies (tests etc.)
pip install "lora-pytorch[all] @ git+ssh://git@github.com/fkodom/lora-pytorch.git"
# Setup pre-commit hooks
pre-commit install
```


## Usage

How to use LoRA with your PyTorch model:
```python
import torch
from lora_pytorch import LoRA
from torchvision.models import resnet18, ResNet

# Wrap your model with LoRA
model = resnet18()
lora_model = LoRA.from_module(model, rank=5)

print(lora_model)
# LoRA(
#   (module): ResNet(
#     (conv1): LoRA(
#       (module): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#       (lora_module): Conv2dLoRAModule(
#         (in_conv): Conv2d(3, 5, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#         (out_conv): Conv2d(5, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (dropout): Dropout(p=0.0, inplace=False)
#       )
#     )
#     (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (relu): ReLU(inplace=True)
# ...

# Train or predict as usual.
x = torch.randn(1, 3, 224, 224)
y = lora_model(x)
# compute loss, backprop, etc...

# Merge LoRA weights into the original model.
new_model = lora_model.merge_lora(inplace=False)  # default: inplace=False

# NOTE: new_model has the same type as the original model!  Inference is just as
# fast as in the original model.
assert isinstance(new_model, ResNet)
```

### Advanced Usage

Enable or disable `LoRA` as needed. (e.g. to access the original model)

**NOTE**: `LoRA` will *not* track gradients from the original model.
```python
# Disable
lora_model.disable_lora()
y = lora_model(x)
print(y.requires_grad)
# False

# Re-enable
lora_model.enable_lora()
y = lora_model(x)
print(y.requires_grad)
# True
```

Remove `LoRA` from the model.

**NOTE**: The original model weights will be unchanged.
```python
# Remove
original_model = lora_model.remove_lora(inplace=False)  # default: inplace=False
assert isinstance(original_model, ResNet)
```


## Supported Layers

I hope to add support for more layer types.  Specifically, all of the layers listed below will eventually be supported.  If you need support for a specific layer, please open an issue or a pull request.

Layer | Supported
--- | ---
`nn.Linear` | ✅
`nn.MultiheadAttention` | ✅
`nn.Conv1d` | ✅
`nn.Conv2d` | ✅
`nn.Conv3d` | ✅
`nn.ConvTranspose1d` | ❌
`nn.ConvTranspose2d` | ❌
`nn.ConvTranspose3d` | ❌

**NOTE**: Activation, normalization, dropout, etc. layers are not affected by `LoRA`.  Those are not listed here, but you shouldn't have any problems using them.


## TODO

* Add support for more layer types (see above)
* Experiments with large, pretrained models
    * Specifically, models that are not covered by LoRA in [huggingface/transformers](https://github.com/huggingface/transformers), for example.
    * Lots of CV examples: ResNet, ViT, DETR, UNET, DeepLab, etc.

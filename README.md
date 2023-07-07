# lora-pytorch

## Install

```bash
pip install "lora-pytorch @ git+ssh://git@github.com/fkodom/lora-pytorch.git"

# Install all dev dependencies (tests etc.)
pip install "lora-pytorch[all] @ git+ssh://git@github.com/fkodom/lora-pytorch.git"

# Setup pre-commit hooks
pre-commit install
```


## Test

Tests run automatically through GitHub Actions.
* Fast tests run on each push.
* Slow tests (decorated with `@pytest.mark.slow`) run on each PR.

You can also run tests manually with `pytest`:
```bash
pytest lora-pytorch

# For all tests, including slow ones:
pytest --slow lora-pytorch
```


## Release

[Optional] Requires either PyPI or Docker GHA workflows to be enabled.

Just tag a new release in this repo, and GHA will automatically publish Python wheels and/or Docker images.

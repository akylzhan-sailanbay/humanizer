def test_package_imports():
    import humanizer

    assert humanizer.__version__ == "0.1.0"


def test_mps_available():
    """The whole design assumes MPS. If this fails, nothing downstream is valid."""
    import torch

    assert torch.backends.mps.is_available()

import sys
import os
import pytest
import torch


@pytest.fixture(scope="session")
def test_dataset():
    return torch.load("tmp/test_dataset.pt", weights_only=False)

@pytest.fixture(scope="session")
def latest_model_path():
    return "tmp/latest_model"

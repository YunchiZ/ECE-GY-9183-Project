import pytest
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset_classes import NewsDataset

@pytest.fixture(scope="session")
def test_dataset():
    # load test_dataset.pt
    dataset = torch.load("tmp/test_dataset.pt", weights_only=False)
    return dataset
 
@pytest.fixture(scope="session")
def test_labels():
    # load test_dataset.pt
    labels = torch.load("tmp/test_labels.pt", weights_only=False)
    return labels

@pytest.fixture(scope="session")
def latest_model_path():
    return "tmp/latest_model"

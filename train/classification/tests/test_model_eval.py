import pytest
from eval_model import evaluate_model_accuracy
from datasets import load_dataset
from transformers import DistilBertForSequenceClassification
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset_classes import NewsDataset

def test_accuracy_score_above_threshold(test_dataset, test_labels, latest_model_path):
    threshold = 0.7

    model = DistilBertForSequenceClassification.from_pretrained(latest_model_path)
    # test_dataset = torch.load(test_dataset)
    # test_labels = torch.load(test_labels)

    accuracy = evaluate_model_accuracy(model, test_dataset, test_labels)

    print(f"Evaluated accuracy: {accuracy:.4f}")
    
    assert accuracy >= threshold, f"Model accuracy score {accuracy:.4f} is below threshold {threshold}"


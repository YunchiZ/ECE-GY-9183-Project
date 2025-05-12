import pytest
from eval_model import evaluate_model_f1
from datasets import load_dataset
from transformers import XLNetForSequenceClassification
import torch

def test_f1_score_above_threshold(test_dataset, latest_model_path):
    threshold = 0.85

    model = XLNetForSequenceClassification.from_pretrained(latest_model_path)

    f1 = evaluate_model_f1(model, test_dataset)

    print(f"Evaluated F1: {f1:.4f}")
    
    assert f1 >= threshold, f"Model F1 score {f1:.4f} is below threshold {threshold}"

import pytest
from eval_model import evaluate_model_f1  # 替换为你保存上面函数的模块
from datasets import load_dataset  # 示例加载
from transformers import XLNetForSequenceClassification
import torch

def test_f1_score_above_threshold():
    threshold = 0.85

    model = XLNetForSequenceClassification.from_pretrained("tmp/latest_model")
    test_dataset = torch.load("tmp/test_dataset.pt")

    f1 = evaluate_model_f1(model, test_dataset)

    print(f"Evaluated F1: {f1:.4f}")
    
    assert f1 >= threshold, f"Model F1 score {f1:.4f} is below threshold {threshold}"

import pytest
from eval_model import evaluate_model_accuracy  # 替换为你保存上面函数的模块
from datasets import load_dataset  # 示例加载
from transformers import DistilBertForSequenceClassification
import torch
from dataset_classes import NewsDataset

def test_accuracy_score_above_threshold():
    threshold = 0.7

    model = DistilBertForSequenceClassification.from_pretrained("tmp/latest_model")
    test_dataset = torch.load("tmp/test_dataset.pt")
    test_labels = torch.load("tmp/test_labels.pt")

    accuracy = evaluate_model_accuracy(model, test_dataset, test_labels)

    print(f"Evaluated accuracy: {accuracy:.4f}")
    
    assert accuracy >= threshold, f"Model accuracy score {accuracy:.4f} is below threshold {threshold}"


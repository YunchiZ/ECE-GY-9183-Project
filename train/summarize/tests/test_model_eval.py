import pytest
from eval_model import evaluate_model_rouge
from datasets import load_dataset
from transformers import BartForConditionalGeneration, BartTokenizer
import torch

def test_rouge_score_above_threshold(test_dataset, latest_model_path):
    threshold = 0.2

    model = BartForConditionalGeneration.from_pretrained(latest_model_path)
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base", cache_dir='./model')
    
    # test_dataset = torch.load("tmp/test_dataset.pt")

    rouge = evaluate_model_rouge(test_dataset, model, tokenizer)

    print(f"Evaluated rougeL: {rouge['rougeL']:.4f}")
    
    assert rouge['rougeL'] >= threshold, f"Model rougeL score {rouge['rougeL']:.4f} is below threshold {threshold}"


from transformers import Trainer, TrainingArguments, XLNetForSequenceClassification
import numpy as np
from sklearn.metrics import accuracy_score
import torch
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from dataset_classes import NewsDataset

        
def evaluate_model_accuracy(model, test_dataset, test_labels):
    # model = XLNetForSequenceClassification.from_pretrained(model_path)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="./tmp"),
    )
    predictions = trainer.predict(test_dataset)
    predictions_logits = predictions.predictions
    predicted_labels = np.argmax(predictions_logits, axis=1)
    accuracy = accuracy_score(test_labels, predicted_labels)
    return accuracy


    


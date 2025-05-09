from transformers import Trainer, TrainingArguments, XLNetForSequenceClassification
import numpy as np
from sklearn.metrics import f1_score

def evaluate_model_f1(model, test_dataset):
    # model = XLNetForSequenceClassification.from_pretrained(model_path)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="./tmp", per_device_eval_batch_size=8),
    )
    predictions = trainer.predict(test_dataset)
    predictions_logits = predictions.predictions
    predicted_labels = np.argmax(predictions_logits, axis=1)
    f1 = f1_score(test_dataset['label'], predicted_labels, average="binary")
    return f1


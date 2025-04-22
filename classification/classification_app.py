import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import wandb
import torch
from sklearn.metrics import accuracy_score, f1_score
import ray
from ray import tune
from ray.air import session
import os
from pathlib import Path
import json
from filelock import FileLock
import pytest
import subprocess
from dataset_classes import NewsDataset



def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Get the predicted class by using argmax (for multi-class classification)
    preds = np.argmax(predictions, axis=1)
    
    # Calculate accuracy and F1 score
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")  # Use 'micro', 'macro', or 'weighted' based on the task

    return {
        'accuracy': accuracy,
        'f1': f1,
    }




def train_fn(config, model, train_dataset, eval_dataset):

    try:
        trial_dir = session.get_trial_dir()  # 例如：~/ray_results/test/trial_xxx/
        output_dir = os.path.join(trial_dir, "results")
    except Exception as e:
        print(f"路径错误: {str(e)}")
        raise
    
    # Update training arguments with the hyperparameters from Ray Tune
    training_args = TrainingArguments(
        run_name = "id_1_epoch_2",
        output_dir=output_dir,
        num_train_epochs=2,  
        
        # per_device_train_batch_size=config["batch_size"],  # Hyperparameter from Ray Tune
        # per_device_eval_batch_size=config["batch_size"],   # Hyperparameter from Ray Tune
        per_device_train_batch_size=16,  # Hyperparameter from Ray Tune
        per_device_eval_batch_size=16,   # Hyperparameter from Ray Tune
        # warmup_steps=config["warmup_steps"],               # Hyperparameter from Ray Tune
        warmup_steps=500,
        learning_rate=config["learning_rate"],              # Hyperparameter from Ray Tune
        
        weight_decay=0.01,
        logging_dir=os.path.join(trial_dir, "logs"),  
        logging_steps=500,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        metric_for_best_model="eval_accuracy",
    )

    
    # Initialize the Trainer
    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset, 
        compute_metrics=compute_metrics,
    )
    try:
        # Train the model
        trainer.train()
    except Exception as e:
        print(f"训练失败: {str(e)}")
        raise

    try:
    # Evaluate the model
        eval_results = trainer.evaluate()
    except Exception as e:
        print(f"评估失败: {str(e)}")
        raise

    try:
    # Return the evaluation results to Ray Tune
        tune.report(metrics=eval_results)
        trainer.save_model(output_dir)
        tune.report(
            metrics=eval_results,
            checkpoint=tune.Checkpoint.from_directory(output_dir)  # 将模型目录作为检查点
        )
    except Exception as e:
        print(f"报告错误: {str(e)}")
        raise


def get_next_model_version(base_dir="model"):
    base_path = Path(base_dir)
    base_path.mkdir(exist_ok=True)  # 创建 model 文件夹（如果不存在）

    existing_versions = [
        int(p.name.replace("BERT-v", ""))
        for p in base_path.iterdir()
        if p.is_dir() and p.name.startswith("BERT-v") and p.name.replace("BERT-v", "").isdigit()
    ]
    next_version = max(existing_versions + [-1]) + 1  # 如果不存在则为 v0
    return f"BERT-v{next_version}", base_path / f"BERT-v{next_version}"


def update_model_status(new_model, new_scores):
    status_path = Path("model_status.json")
    model_status = {}

    # 使用 portalocker 锁定文件读写
    with portalocker.Lock(status_path, timeout=10, mode="a+") as f:
        f.seek(0)
        try:
            model_status = json.load(f)
        except json.JSONDecodeError:
            model_status = {}

        model_status[new_model] = "candidate"

        serving_name = next((k for k, v in model_status.items() if v == "serving"), None)
        serving_path = f"model/{serving_name}"
        serving_model = DistilBertForSequenceClassification.from_pretrained(serving_path)
        
        print(serving_model)

        trainer = Trainer(
            model=serving_model,
            args=TrainingArguments(output_dir="./tmp"),  # 临时目录，仅用于预测
        )
        
        predictions = trainer.predict(test_dataset)
        predictions_logits = predictions.predictions
        predicted_labels = np.argmax(predictions_logits, axis=1)
        
        accuracy = accuracy_score(test_labels_encoded, predicted_labels)
        print(f"old Accuracy: {accuracy:.4f}")

        if new_scores >= accuracy:
            model_status[new_model] = "serving"
            model_status[serving_name] = "candidate"

        f.seek(0)
        f.truncate()
        json.dump(model_status, f, indent=4)

def update_model_status(new_model_name):
    task_id = "1"
    
    status_path = Path("model_status.json")
    lock_path = status_path.with_suffix(".lock")
    lock = FileLock(str(lock_path))
    
    model_status = {}

    # 使用 portalocker 锁定文件读写
    with lock:
        if status_path.exists():
            with status_path.open("r") as f:
                try:
                    model_status = json.load(f)
                except json.JSONDecodeError:
                    model_status = {}
        else:
            model_status = {}

        if task_id not in model_status:
            model_status[task_id] = []

        task_models = model_status[task_id]
        # 移除已有相同名字的模型
        task_models = [m for m in task_models if m["model"] != new_model_name]

        # 加入新模型
        task_models.append({
            "model": new_model_name,
            "status": "candidate"
        })
        model_status[task_id] = task_models

        with status_path.open("w") as f:
            json.dump(model_status, f, indent=4)


if __name__ == '__main__':

    data = pd.read_csv("./dataset/NewsCategorizer.csv")
    train_texts, test_texts, train_labels, test_labels = train_test_split(data['short_description'], data['category'], test_size=0.2, shuffle=True)
    train_texts, eval_texts, train_labels, eval_labels = train_test_split(data['short_description'], data['category'], test_size=0.2, shuffle=True)

    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_labels)
    test_labels_encoded = label_encoder.transform(test_labels)
    eval_labels_encoded = label_encoder.transform(eval_labels)

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased', cache_dir='./model')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(data['category'].unique()), cache_dir='./model')

    train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True)
    test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True)
    eval_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True)

    train_labels_tensor = torch.tensor(train_labels_encoded)
    test_labels_tensor = torch.tensor(test_labels_encoded)
    eval_labels_tensor = torch.tensor(eval_labels_encoded)

    train_dataset = NewsDataset(train_encodings, train_labels_tensor)
    test_dataset = NewsDataset(test_encodings, test_labels_tensor)
    eval_dataset = NewsDataset(eval_encodings, eval_labels_tensor)
    search_space = {
    "learning_rate": tune.grid_search([1e-5]),
    # "batch_size": tune.choice([8, 16]),
    # "warmup_steps": tune.choice([500, 1000, 2000]),
    }

    wandb.init(project="Mlops-classification", entity="yunchiz-new-york-university")

    current_dir = os.getcwd()
    storage_path = f"file://{current_dir}/ray_results"

    train_fn_with_params = tune.with_parameters(train_fn, model=model, train_dataset=train_dataset, eval_dataset=eval_dataset)
    ray.init(ignore_reinit_error=True)  # Initialize Ray
    analysis = tune.run(
        train_fn_with_params,  # The training function that Ray Tune will use
        config=search_space,  # The search space of hyperparameters
        # resources_per_trial={"cpu": 1, "gpu": 1},
        resources_per_trial={"cpu": 0, "gpu": 1},
        num_samples=1,  # Number of trials (hyperparameter combinations)
        verbose=1,  # Verbosity level of Ray Tune
        storage_path=storage_path,
        name="id_1_epoch_2",
    )

    best_trial = analysis.get_best_trial(metric="eval_accuracy", mode="max")

    # 获取检查点路径（通过 checkpoint 属性）
    best_checkpoint = best_trial.checkpoint
    best_checkpoint_dir = best_checkpoint.to_directory()
    print(f"最佳模型路径：{best_checkpoint_dir}")

    best_model = DistilBertForSequenceClassification.from_pretrained(best_checkpoint_dir)
    
    best_model.save_pretrained("tmp/latest_model")
    torch.save(test_dataset, "tmp/test_dataset.pt")
    torch.save(test_labels_encoded, "tmp/test_labels.pt")
    
    retcode = subprocess.call(["pytest", "-v", "-s", "test_model_eval.py"])
    if retcode != 0:
        print("test failed")
    else:
        model_name, save_path = get_next_model_version()
        best_model.save_pretrained(save_path)
        print(f"new model: {model_name} + {save_path}")
        update_model_status(model_name)
    wandb.finish()



    
import wandb
import pandas as pd
from datasets import Dataset
from datasets import DatasetDict
from sklearn.model_selection import train_test_split
from transformers import XLNetTokenizer
import sentencepiece
from transformers import XLNetForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch

import os
os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"

from ray.air import session
import ray
from ray import tune
from pathlib import Path
import json
from filelock import FileLock
import pytest
import subprocess
from datetime import datetime


def preprocess(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=256,
        return_tensors="pt"
    )



def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Get the predicted class by using argmax (for multi-class classification)
    preds = np.argmax(predictions, axis=1)
    
    # Calculate accuracy and F1 score
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="binary")  # Use 'micro', 'macro', or 'weighted' based on the task

    return {
        'accuracy': accuracy,
        'f1': f1,
    }



def train_fn(config, model, train_dataset, eval_dataset, run_name):
    trial_id = session.get_trial_name()
    wandb.init(
        project="Mlops-fakenews",
        entity="yunchiz-new-york-university",
        name=f"{run_name}_{trial_id}",
        group=run_name,  # 所有 trial 同组
        reinit=True
    )

    try:
        trial_dir = session.get_trial_dir()  # 例如：~/ray_results/test/trial_xxx/
        output_dir = os.path.join(trial_dir, "results")
    except Exception as e:
        print(f"路径错误: {str(e)}")
        raise
    
    # Update training arguments with the hyperparameters from Ray Tune
    training_args = TrainingArguments(
        run_name=None,
        output_dir=output_dir,
        num_train_epochs=1,  
        
        # per_device_train_batch_size=config["batch_size"],  # Hyperparameter from Ray Tune
        # per_device_eval_batch_size=config["batch_size"],   # Hyperparameter from Ray Tune
        per_device_train_batch_size=16,  # Hyperparameter from Ray Tune
        per_device_eval_batch_size=16,   # Hyperparameter from Ray Tune
        # warmup_steps=config["warmup_steps"],               # Hyperparameter from Ray Tune
        warmup_steps=500,
        learning_rate=config["learning_rate"],              # Hyperparameter from Ray Tune
        # learning_rate=1e-5,
        
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
        trainer.save_model(output_dir)
        tune.report(
            metrics=eval_results,
            checkpoint=tune.Checkpoint.from_directory(output_dir)  # 将模型目录作为检查点
        )
        wandb.log({
            "trial_learning_rate": config["learning_rate"],
            "trial_accuracy": eval_results["eval_accuracy"],
            "trial_f1": eval_results["eval_f1"],
        })
    except Exception as e:
        print(f"报告错误: {str(e)}")
        raise

def get_next_model_version(base_dir="model"):
    base_path = Path(base_dir)
    base_path.mkdir(exist_ok=True)  # 创建 model 文件夹（如果不存在）

    existing_versions = [
        int(p.name.replace("XLN-v", ""))
        for p in base_path.iterdir()
        if p.is_dir() and p.name.startswith("XLN-v") and p.name.replace("XLN-v", "").isdigit()
    ]
    next_version = max(existing_versions + [-1]) + 1  # 如果不存在则为 v0
    return f"XLN-v{next_version}", base_path / f"XLN-v{next_version}"

def update_model_status(new_model_name):
    task_id = "1"
    
    current_dir = Path(__file__).resolve().parent
    parent_dir = current_dir.parent
    status_path = parent_dir / "model_status.json"
    
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

def evaluate_offline():
    retcode = pytest.main([
        "tests",           # 只扫描 tests/
        "--disable-warnings",
        "-v",                 # (可选) verbose，显示详细每个测试
        "-s"                  # (可选) 允许 print() 打印到控制台
    ])
    return retcode

if __name__ == '__main__':
    df = pd.read_csv("./dataset/WELFake_Dataset.csv")
    df = df.dropna()
    df['text'] = df['title'] + " " + df['text']


    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, eval_df = train_test_split(train_df, test_size=0.2, random_state=42)


    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', cache_dir='./model')
    train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
    eval_dataset = Dataset.from_pandas(eval_df[['text', 'label']])
    test_dataset = Dataset.from_pandas(test_df[['text', 'label']])

    train_dataset = train_dataset.map(preprocess, batched=True)
    eval_dataset = eval_dataset.map(preprocess, batched=True)
    test_dataset = test_dataset.map(preprocess, batched=True)

    model = XLNetForSequenceClassification.from_pretrained(
        'xlnet/xlnet-base-cased',
        num_labels=2,
        problem_type="single_label_classification",
        cache_dir='./model'
    )

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to = "none"
    )

    search_space = {
        "learning_rate": tune.grid_search([1e-5, 2e-5]),
        # "batch_size": tune.choice([8, 16]),
        # "warmup_steps": tune.choice([500, 1000, 2000]),
    }

    model_name, save_path = get_next_model_version()
    run_name = f"{model_name}_{datetime.now().strftime('%m%d_%H%M')}"
    # wandb.init(project="Mlops-fakenews", entity="yunchiz-new-york-university", name=run_name)
    os.environ["WANDB_PROJECT"] = "Mlops-fakenews"
    os.environ["WANDB_DISABLED"] = "false"
    
    current_dir = os.getcwd()
    storage_path = f"file://{current_dir}/ray_results"

    train_fn_with_params = tune.with_parameters(train_fn, model=model, train_dataset=train_dataset, eval_dataset=eval_dataset, run_name = run_name)
    ray.init(ignore_reinit_error=True)  # Initialize Ray
    analysis = tune.run(
        train_fn_with_params,  # The training function that Ray Tune will use
        config=search_space,  # The search space of hyperparameters
        # resources_per_trial={"cpu": 1, "gpu": 1},
        resources_per_trial={"cpu": 0, "gpu": 1},
        num_samples=1,  # Number of trials (hyperparameter combinations)
        verbose=1,  # Verbosity level of Ray Tune
        storage_path=storage_path,
        name=model_name,
        callbacks=[],
    )

    best_trial = analysis.get_best_trial(metric="eval_accuracy", mode="max")

    # 获取检查点路径（通过 checkpoint 属性）
    best_checkpoint = best_trial.checkpoint
    best_checkpoint_dir = best_checkpoint.to_directory()  # 提取检查点目录
    print(f"最佳模型路径：{best_checkpoint_dir}")
    best_model = XLNetForSequenceClassification.from_pretrained(best_checkpoint_dir)
    best_model.save_pretrained("tmp/latest_model")

    torch.save(test_dataset, "tmp/test_dataset.pt")
    # artifact = wandb.Artifact(name=f"{run_name}_best_model", type="model")
    # artifact.add_dir(str("tmp/latest_model"))
    # wandb.log_artifact(artifact)

    retcode = evaluate_offline()

    if retcode != 0:
        print("test failed")
    else:
        best_model.save_pretrained(save_path)
        print(f"new model: {model_name} + {save_path}")
        update_model_status(model_name)
        
    
    wandb.finish()

    
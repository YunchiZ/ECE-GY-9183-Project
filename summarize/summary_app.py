from datasets import load_dataset
from transformers import BartTokenizer, BartForConditionalGeneration, TrainingArguments, Trainer
import evaluate
import numpy as np
import wandb
import datasets
from ray import tune
import ray
import os
from ray.air import session
from sklearn.metrics import accuracy_score, f1_score
import torch
import subprocess
from pathlib import Path
import json
from filelock import FileLock
import pytest
import subprocess

max_input_length = 1024
max_target_length = 128



def preprocess_function(examples):
    inputs = examples["article"]
    targets = examples["highlights"]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=max_target_length, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    
    # 显式保留原始文本列
    model_inputs["article"] = examples["article"]
    model_inputs["highlights"] = examples["highlights"]
    return model_inputs


# import evaluate
def compute_metrics(eval_pred):
    try:
        rouge = evaluate.load('rouge')
        
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
        
        # Handle case where predictions might be logits for seq2seq models
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        # Check if predictions are already token IDs or if they're logits
        if len(predictions.shape) > 2:  # For 3D tensor, likely [batch, seq, vocab]
            predictions = np.argmax(predictions, axis=-1)
        
        # Decode token IDs to text
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Clean up
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        
        # Calculate ROUGE
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        
        # Different Rouge libraries return scores in different formats
        # Some return direct float values, others return objects with mid.fmeasure
        rouge_scores = {}
        
        # Check the format of result and extract scores accordingly
        for metric, value in result.items():
            if hasattr(value, 'mid') and hasattr(value.mid, 'fmeasure'):
                # This is the format where scores are objects with mid.fmeasure
                rouge_scores[metric] = value.mid.fmeasure
            elif isinstance(value, (float, np.float64, np.float32)):
                # Direct float values
                rouge_scores[metric] = float(value)
            else:
                # Fallback - convert whatever value we have to float if possible
                try:
                    rouge_scores[metric] = float(value)
                except:
                    print(f"Warning: Could not convert {metric} score to float, using 0.0")
                    rouge_scores[metric] = 0.0
        
        # Add generation length metric
        rouge_scores["gen_len"] = np.mean([len(pred.split()) for pred in decoded_preds])
        
        return rouge_scores
        
    except Exception as e:
        print(f"计算指标时出错：{e}")
        # Return default values to avoid interrupting training
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "gen_len": 0.0}




def train_fn(config, model, train_dataset, eval_dataset):
    try:
        trial_dir = session.get_trial_dir()  # 例如：~/ray_results/test/trial_xxx/
        output_dir = os.path.join(trial_dir, "results")
    except Exception as e:
        print(f"路径错误: {str(e)}")
        raise


    training_args = TrainingArguments(
        run_name = "ray_test_epoch_2",
        output_dir=output_dir,
        num_train_epochs=2,  
        
        # per_device_train_batch_size=config["batch_size"],  # Hyperparameter from Ray Tune
        # per_device_eval_batch_size=config["batch_size"],   # Hyperparameter from Ray Tune
        # gradient_accumulation_steps=config["gradient_accumulation_steps"],               # Hyperparameter from Ray Tune

        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        
        learning_rate=config["learning_rate"],              # Hyperparameter from Ray Tune
        
        weight_decay=0.01,
        logging_dir=os.path.join(trial_dir, "logs"),  
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",

        save_total_limit=3,
        metric_for_best_model="rougeL",
        fp16=True,
    )

    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=train_subset, 
        eval_dataset=eval_subset, 
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
        int(p.name.replace("BART-v", ""))
        for p in base_path.iterdir()
        if p.is_dir() and p.name.startswith("BART-v") and p.name.replace("BART-v", "").isdigit()
    ]
    next_version = max(existing_versions + [-1]) + 1  # 如果不存在则为 v0
    return f"BART-v{next_version}", base_path / f"BART-v{next_version}"



def update_model_status(new_model_name):
    task_id = "0"
    
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

    save_path = './dataset'
    dataset = load_dataset('abisee/cnn_dailymail', '3.0.0', cache_dir=save_path)

    model_name = "facebook/bart-base"
    tokenizer = BartTokenizer.from_pretrained(model_name, cache_dir='./model')
    model = BartForConditionalGeneration.from_pretrained(model_name, cache_dir='./model')


    ## autodl vpn
    result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
    output = result.stdout
    for line in output.splitlines():
        if '=' in line:
            var, value = line.split('=', 1)
            os.environ[var] = value

    
    # 冻结编码器部分的所有层
    for param in model.model.encoder.parameters():
        param.requires_grad = False

    # 仅训练解码器部分的层
    for param in model.model.decoder.parameters():
        param.requires_grad = True

    # 应用预处理函数时，仅移除不需要的列（如果有）
    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=[]
    )

    train_size = 1000  # You can adjust this number to use fewer examples
    eval_size = 200  # Same for evaluation dataset

    train_subset = tokenized_datasets["train"].select(range(train_size))
    eval_subset = tokenized_datasets["validation"].select(range(eval_size))

    search_space = {
        # "learning_rate": tune.grid_search([1e-5, 2e-5, 5e-5]),
        # "batch_size": tune.choice([8, 16]),
        # "warmup_steps": tune.choice([500, 1000, 2000]),
        "learning_rate": tune.grid_search([1e-5]),
    }

    wandb.init(project="Mlops-summary", entity="yunchiz-new-york-university")

    current_dir = os.getcwd()
    storage_path = f"file://{current_dir}/ray_results"

    train_fn_with_params = tune.with_parameters(train_fn, model=model, train_dataset=train_subset, eval_dataset=eval_subset)
    ray.init(ignore_reinit_error=True)  # Initialize Ray
    analysis = tune.run(
        train_fn_with_params,  # The training function that Ray Tune will use
        config=search_space,  # The search space of hyperparameters
        # resources_per_trial={"cpu": 1, "gpu": 1},
        resources_per_trial={"cpu": 0, "gpu": 1},
        num_samples=1,  # Number of trials (hyperparameter combinations)
        verbose=1,  # Verbosity level of Ray Tune
        storage_path=storage_path,
        name="ray_test_epoch_2",
    )

    test_size = 200  # Same for evaluation dataset
    test_dataset = tokenized_datasets["test"].select(range(test_size))

    best_trial = analysis.get_best_trial(metric="eval_accuracy", mode="max")

    # 获取检查点路径（通过 checkpoint 属性）
    best_checkpoint = best_trial.checkpoint
    best_checkpoint_dir = best_checkpoint.to_directory()
    print(f"最佳模型路径：{best_checkpoint_dir}")

    # 加载模型
    from transformers import AutoModel
    best_model = BartForConditionalGeneration.from_pretrained(best_checkpoint_dir)

    best_model.save_pretrained("tmp/latest_model")
    torch.save(test_dataset, "tmp/test_dataset.pt")
    retcode = subprocess.call(["pytest", "-v", "-s", "test_model_eval.py"])
    if retcode != 0:
        print("test failed")
    else:
        model_name, save_path = get_next_model_version()
        best_model.save_pretrained(save_path)
        print(f"new model: {model_name} + {save_path}")
        update_model_status(model_name)
    wandb.finish()

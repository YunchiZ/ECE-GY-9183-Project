from datasets import load_dataset
from transformers import BartTokenizer, BartForConditionalGeneration, TrainingArguments, Trainer
import evaluate
import numpy as np
import wandb
import datasets
import os
os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
from ray import tune
import ray
from ray.air import session
from sklearn.metrics import accuracy_score, f1_score
import torch
import subprocess
from pathlib import Path
import json
from filelock import FileLock
import pytest
import subprocess
from datetime import datetime
from peft import get_peft_model, LoraConfig, TaskType
from transformers.onnx import export
from transformers.onnx.features import FeaturesManager

max_input_length = 1024
max_target_length = 128



def preprocess_function(examples):
    inputs = examples["article"]
    targets = examples["highlights"]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=max_target_length, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    
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
        if len(predictions.shape) > 2:  # For 3D tensor
            predictions = np.argmax(predictions, axis=-1)
        
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        rouge_scores = {}
        
        # Check the format of result and extract scores accordingly
        for metric, value in result.items():
            if hasattr(value, 'mid') and hasattr(value.mid, 'fmeasure'):
                rouge_scores[metric] = value.mid.fmeasure
            elif isinstance(value, (float, np.float64, np.float32)):
                rouge_scores[metric] = float(value)
            else:
                try:
                    rouge_scores[metric] = float(value)
                except:
                    print(f"Warning: Could not convert {metric} score to float, using 0.0")
                    rouge_scores[metric] = 0.0
        
        # generation length
        rouge_scores["gen_len"] = np.mean([len(pred.split()) for pred in decoded_preds])
        
        return rouge_scores
        
    except Exception as e:
        print(f"metrics error：{e}")
        # Return default values
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "gen_len": 0.0}


def train_fn(config, model, train_dataset, eval_dataset, run_name):
    trial_id = session.get_trial_name()
    wandb.init(
        project="Mlops-summary",
        entity="yunchiz-new-york-university",
        name=f"{run_name}_{trial_id}",
        group=run_name,
        reinit=True
    )
    try:
        trial_dir = session.get_trial_dir()
        output_dir = os.path.join(trial_dir, "results")
    except Exception as e:
        print(f"路径错误: {str(e)}")
        raise


    training_args = TrainingArguments(
        run_name=None,
        output_dir=output_dir,
        num_train_epochs=1,  

        per_device_train_batch_size=8,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        
        learning_rate=config["learning_rate"],  # Hyperparameter from Ray Tune
        
        weight_decay=0.01,
        logging_dir=os.path.join(trial_dir, "logs"),  
        logging_steps=10000,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=1,
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
        # Train
        trainer.train()
    except Exception as e:
        print(f"Train fail: {str(e)}")
        raise

    try:
        # Evaluate
        eval_results = trainer.evaluate()
    except Exception as e:
        print(f"Eval fail: {str(e)}")
        raise

    try:
    # Return the evaluation results to Ray Tune and Wandb
        tune.report(metrics=eval_results)
        trainer.save_model(output_dir)
        tune.report(
            metrics=eval_results,
            checkpoint=tune.Checkpoint.from_directory(output_dir)
        )
        wandb.log({
            "trial_learning_rate": config["learning_rate"],
            "rouge1": eval_results["eval_rouge1"],
            "rouge2": eval_results["eval_rouge2"],
            "rougeL": eval_results["eval_rougeL"],
        })
    except Exception as e:
        print(f"Log fail: {str(e)}")
        raise
        
def get_next_model_version(base_dir="model"):
    task_id = "0"

    current_dir = Path(__file__).resolve().parent
    parent_dir = current_dir.parent
    status_path = parent_dir / "model/model_status.json"
    
    lock_path = status_path.with_suffix(".lock")
    lock = FileLock(str(lock_path))
    with lock:
        if not status_path.exists():
            return None
        try:
            with status_path.open("r") as f:
                model_status = json.load(f)
        except json.JSONDecodeError:
            return None

        if task_id not in model_status or not model_status[task_id]:
            return None

        versions = []
        for entry in model_status[task_id]:
            model_name = entry.get("model", "")
            if model_name.startswith("BART-v"):
                try:
                    version_num = int(model_name.replace("BART-v", ""))
                    versions.append(version_num)
                except ValueError:
                    continue

        return max(versions) if versions else 0
    # return f"BART-v{next_version}", base_path / f"BART-v{next_version}"



def update_model_status(new_model_name):
    task_id = "0"
    
    current_dir = Path(__file__).resolve().parent
    parent_dir = current_dir.parent
    status_path = parent_dir / "model/model_status.json"
    
    lock_path = status_path.with_suffix(".lock")
    lock = FileLock(str(lock_path))
    model_status = {}

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
        # remove model with same name
        task_models = [m for m in task_models if m["model"] != new_model_name]

        # add new model
        task_models.append({
            "model": new_model_name,
            "status": "candidate"
        })
        model_status[task_id] = task_models

        with status_path.open("w") as f:
            json.dump(model_status, f, indent=4)

def evaluate_offline():
    retcode = pytest.main([
        "./app/tests/bart_test",        # tests/
        "--disable-warnings",
        "-v",
        "-s"
    ])
    return retcode

def export_model_to_onnx(model, tokenizer, output_path, model_name):
    # Fix output path - ensure it points to a file, not a directory
    if os.path.isdir(output_path):
        output_file = os.path.join(output_path, f"{model_name}.onnx")
    else:
        output_file = output_path
        
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"Will export to: {output_file}")
    
    # Try all methods in order until one succeeds
    methods = [
        export_with_transformers_api,
        export_with_direct_torch
    ]
    
    last_error = None
    for method in methods:
        try:
            print(f"Trying export method: {method.__name__}")
            result = method(model, tokenizer, output_file)
            if result:
                return result
        except Exception as e:
            print(f"Method {method.__name__} failed: {e}")
            last_error = e
            import traceback
            traceback.print_exc()
    
    # All methods failed
    raise RuntimeError(f"All ONNX export methods failed. Last error: {last_error}")


def export_with_transformers_api(model, tokenizer, output_file):
    """Try exporting with the transformers ONNX API"""
    try:
        from transformers.onnx import export
        from transformers.onnx.features import FeaturesManager
        from pathlib import Path
        
        # Set model to evaluation mode
        model.eval()
        
        # Get model type
        model_type = model.config.model_type
        
        # Check which methods are available in FeaturesManager
        print("Available FeaturesManager methods:")
        for method in dir(FeaturesManager):
            if not method.startswith("_"):
                print(f"- {method}")
        
        # Try the get_config method (used in newer versions)
        if hasattr(FeaturesManager, "get_config"):
            onnx_config = FeaturesManager.get_config(
                model_type=model_type,
                task="seq2seq-lm"
            )
            
            export(
                tokenizer=tokenizer,
                model=model,
                config=onnx_config,
                opset=14,
                output=Path(output_file)
            )
            
            return output_file
            
        # If the above doesn't work, try alternative approaches
        print("Trying alternative transformers export approaches...")
        return None
        
    except Exception as e:
        print(f"Transformers API export failed: {e}")
        return None


def export_with_direct_torch(model, tokenizer, output_file):
    """Export the model directly using torch.onnx"""
    try:
        # Set the model to evaluation mode
        model.eval()
        
        # Create sample inputs
        sample_text = "This is a sample text for ONNX export."
        inputs = tokenizer(sample_text, return_tensors="pt")
        
        # For seq2seq models, we need decoder inputs
        decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]])
        
        # Define dynamic axes for variable input sizes
        dynamic_axes = {
            'input_ids': {0: 'batch_size', 1: 'sequence'},
            'attention_mask': {0: 'batch_size', 1: 'sequence'},
            'decoder_input_ids': {0: 'batch_size', 1: 'decoder_sequence'}
        }
        
        # Export directly using torch.onnx
        with torch.no_grad():
            torch.onnx.export(
                model,
                (
                    inputs.input_ids,
                    inputs.attention_mask,
                    decoder_input_ids
                ),
                output_file,
                input_names=['input_ids', 'attention_mask', 'decoder_input_ids'],
                output_names=['logits'],
                dynamic_axes=dynamic_axes,
                opset_version=14,
                do_constant_folding=True,
                export_params=True
            )
        
        return output_file
        
    except Exception as e:
        print(f"Direct torch export failed: {e}")
        return None

if __name__ == '__main__':
    train_dir = Path(__file__).resolve().parent.parent

    # save_path = './dataset'
    # data_path = train_dir / "model/bart_source"
    save_path = train_dir / "model/bart_source"

    dataset = load_dataset('abisee/cnn_dailymail', '3.0.0', cache_dir=save_path)

    model_name = "facebook/bart-base"
    tokenizer = BartTokenizer.from_pretrained(model_name, cache_dir=save_path)
    model = BartForConditionalGeneration.from_pretrained(model_name, cache_dir=save_path)
    
    # 冻结编码器部分的所有层
    for param in model.model.encoder.parameters():
        param.requires_grad = False

    # 仅训练解码器部分的层
    for param in model.model.decoder.parameters():
        param.requires_grad = True

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    
    model = get_peft_model(model, lora_config)

    # 应用预处理函数时，仅移除不需要的列（如果有）
    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=[]
    )

    train_size = 10000  # use fewer examples
    eval_size = 200

    train_subset = tokenized_datasets["train"].select(range(train_size))
    eval_subset = tokenized_datasets["validation"].select(range(eval_size))

    # train_subset = tokenized_datasets["train"]
    # eval_subset = tokenized_datasets["validation"]
    
    search_space = {
        # "batch_size": tune.choice([8, 16]),
        # "warmup_steps": tune.choice([500, 1000, 2000]),
        "learning_rate": tune.grid_search([1e-5, 2e-5]),
    }

    model_id = get_next_model_version() + 1
    
    
    save_path = train_dir / f"model/BART/{model_id}"
    
    save_path.mkdir(parents=True, exist_ok=True)
    model_name = f"BART-v{model_id}"
    torch_path = train_dir / f"model/bart_pytorch/{model_name}"
    
    
    run_name = f"{model_name}_{datetime.now().strftime('%m%d_%H%M')}"
    os.environ["WANDB_PROJECT"] = "Mlops-summary"
    os.environ["WANDB_DISABLED"] = "false"

    current_dir = os.getcwd()
    storage_path = f"file://{current_dir}/ray_results/summary_results"

    train_fn_with_params = tune.with_parameters(train_fn, model=model, train_dataset=train_subset, eval_dataset=eval_subset, run_name = run_name)
    ray.init(_temp_dir=f"{train_dir}/ray_tmp", ignore_reinit_error=True)  # Initialize Ray
    analysis = tune.run(
        train_fn_with_params,
        config=search_space,
        resources_per_trial={"cpu": 0, "gpu": 1},
        num_samples=1,  # Number of trials (hyperparameter combinations)
        verbose=1,
        storage_path=storage_path,
        callbacks=[],
    )

    test_size = 200
    test_dataset = tokenized_datasets["test"].select(range(test_size))

    best_trial = analysis.get_best_trial(metric="eval_rougeL", mode="max")

    best_checkpoint = best_trial.checkpoint
    best_checkpoint_dir = best_checkpoint.to_directory()

    best_model = BartForConditionalGeneration.from_pretrained(best_checkpoint_dir)

    best_model.save_pretrained("tmp/latest_model")
    torch.save(test_dataset, "tmp/test_dataset.pt")
    retcode = evaluate_offline()
    if retcode != 0:
        print("test failed")
    else:
        best_model.save_pretrained(torch_path)
        onnx_path = export_model_to_onnx(best_model, tokenizer, save_path, model_name)
        print(f"New model exported: {model_name}, path: {onnx_path}")
        # print(f"new model: {model_name} + {save_path}")
        update_model_status(model_name)
        print(model_name)
    wandb.finish()

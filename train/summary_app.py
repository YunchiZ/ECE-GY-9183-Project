from datasets import load_dataset
from transformers import BartTokenizer, BartForConditionalGeneration, TrainingArguments, Trainer
import evaluate
import numpy as np
import wandb
import datasets
import os
import logging
os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_DIR"] = "./wandb_results"
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
from functools import partial

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

max_input_length = 1024
max_target_length = 128

def preprocess_function(examples, tokenizer):
    inputs = examples["article"]
    targets = examples["highlights"]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=max_target_length, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    model_inputs["article"] = examples["article"]
    model_inputs["highlights"] = examples["highlights"]
    return model_inputs

def compute_metrics(eval_pred):
    try:
        rouge = evaluate.load('rouge')
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        if len(predictions.shape) > 2:
            predictions = np.argmax(predictions, axis=-1)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        rouge_scores = {}
        for metric, value in result.items():
            if hasattr(value, 'mid') and hasattr(value.mid, 'fmeasure'):
                rouge_scores[metric] = value.mid.fmeasure
            elif isinstance(value, (float, np.float64, np.float32)):
                rouge_scores[metric] = float(value)
            else:
                try:
                    rouge_scores[metric] = float(value)
                except:
                    logger.warning(f"Warning: Could not convert {metric} score to float, using 0.0")
                    rouge_scores[metric] = 0.0
        rouge_scores["gen_len"] = np.mean([len(pred.split()) for pred in decoded_preds])
        return rouge_scores
    except Exception as e:
        logger.error(f"metrics error：{e}")
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
        logger.error(f"路径错误: {str(e)}")
        raise

    training_args = TrainingArguments(
        run_name=None,
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=config["learning_rate"],
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
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Train fail: {str(e)}")
        raise

    try:
        eval_results = trainer.evaluate()
    except Exception as e:
        logger.error(f"Eval fail: {str(e)}")
        raise

    try:
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
        logger.error(f"Log fail: {str(e)}")
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
        task_models = [m for m in task_models if m["model"] != new_model_name]
        task_models.append({"model": new_model_name, "status": "candidate"})
        model_status[task_id] = task_models
        with status_path.open("w") as f:
            json.dump(model_status, f, indent=4)

def evaluate_offline():
    retcode = pytest.main([
        "tests",
        "--disable-warnings",
        "-v",
        "-s"
    ])
    return retcode

def export_model_to_onnx(model, tokenizer, output_path, model_name):
    if os.path.isdir(output_path):
        output_file = os.path.join(output_path, f"{model_name}.onnx")
    else:
        output_file = output_path
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    logger.info(f"Will export to: {output_file}")
    methods = [
        export_with_transformers_api,
        export_with_direct_torch
    ]
    last_error = None
    for method in methods:
        try:
            logger.info(f"Trying export method: {method.__name__}")
            result = method(model, tokenizer, output_file)
            if result:
                return result
        except Exception as e:
            logger.error(f"Method {method.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            last_error = e
    raise RuntimeError(f"All ONNX export methods failed. Last error: {last_error}")

def export_with_transformers_api(model, tokenizer, output_file):
    try:
        from transformers.onnx import export
        from transformers.onnx.features import FeaturesManager
        from pathlib import Path
        model.eval()
        model_type = model.config.model_type
        logger.info("Available FeaturesManager methods:")
        for method in dir(FeaturesManager):
            if not method.startswith("_"):
                logger.info(f"- {method}")
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
        logger.info("Trying alternative transformers export approaches...")
        return None
    except Exception as e:
        logger.error(f"Transformers API export failed: {e}")
        return None

def export_with_direct_torch(model, tokenizer, output_file):
    try:
        model.eval()
        sample_text = "This is a sample text for ONNX export."
        inputs = tokenizer(sample_text, return_tensors="pt")
        decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]])
        dynamic_axes = {
            'input_ids': {0: 'batch_size', 1: 'sequence'},
            'attention_mask': {0: 'batch_size', 1: 'sequence'},
            'decoder_input_ids': {0: 'batch_size', 1: 'decoder_sequence'}
        }
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
        logger.error(f"Direct torch export failed: {e}")
        return None

def summary_run():
    wandb.login(key="<your_wandb_key>")
            
    train_dir = Path(__file__).resolve().parent.parent
    save_path = train_dir / "model"
    dataset = load_dataset('abisee/cnn_dailymail', '3.0.0', cache_dir="./dataset")
    model_name = "facebook/bart-base"
    tokenizer = BartTokenizer.from_pretrained(model_name, cache_dir="./model")
    
    model = BartForConditionalGeneration.from_pretrained(model_name, cache_dir="./model")
    for param in model.model.encoder.parameters():
        param.requires_grad = False
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

    tokenize_fn = partial(preprocess_function, tokenizer=tokenizer)
    
    train_raw = dataset["train"].select(range(100))
    eval_raw = dataset["validation"].select(range(20))
    test_raw = dataset["test"].select(range(20))
    train_subset = train_raw.map(tokenize_fn, batched=True)
    eval_subset = eval_raw.map(tokenize_fn, batched=True)
    test_dataset = test_raw.map(tokenize_fn, batched=True)
    
    search_space = {
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
    ray.init(_temp_dir=f"{train_dir}/ray_tmp", ignore_reinit_error=True)
    analysis = tune.run(
        train_fn_with_params,
        config=search_space,
        resources_per_trial={"cpu": 0, "gpu": 1},
        num_samples=1,
        verbose=1,
        storage_path=storage_path,
        callbacks=[],
    )
    best_trial = analysis.get_best_trial(metric="eval_rougeL", mode="max")
    best_checkpoint = best_trial.checkpoint
    best_checkpoint_dir = best_checkpoint.to_directory()
    best_model = BartForConditionalGeneration.from_pretrained(best_checkpoint_dir)
    best_model.save_pretrained("tmp/latest_model")
    torch.save(test_dataset, "tmp/test_dataset.pt")
    retcode = evaluate_offline()
    if retcode != 0:
        logger.warning("test failed")
    else:
        best_model.save_pretrained(torch_path)
        onnx_path = export_model_to_onnx(best_model, tokenizer, save_path, model_name)
        logger.info(f"New model exported: {model_name}, path: {onnx_path}")
        logger.info(model_name)
    wandb.finish()
    return onnx_path, model_name

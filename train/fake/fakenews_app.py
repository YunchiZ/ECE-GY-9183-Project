# XLNet fake news code with logging
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
import re
import logging
os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
# os.environ["WANDB_MODE"] = "offline"
# os.environ["WANDB_DIR"] = "./wandb_fakenews"
from ray.air import session
import ray
from ray import tune
from pathlib import Path
import json
from filelock import FileLock
import pytest
import subprocess
from datetime import datetime
from transformers.onnx import export
from transformers.onnx.features import FeaturesManager

logging.basicConfig(level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='fakenews_train.log',
    filemode='w' )
logger = logging.getLogger(__name__)

def preprocess(examples, tokenizer):
    return tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=256,
        return_tensors="pt"
    )

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="binary")
    return {
        'accuracy': accuracy,
        'f1': f1,
    }

def train_fn(config, model, train_dataset, eval_dataset, run_name):
    try:
        wandb.finish()
    except:
        pass

    trial_id = session.get_trial_name()
    wandb.init(
        project="Mlops-fakenews",
        entity="yunchiz-new-york-university",
        name=f"{run_name}_{trial_id}",
        group=run_name,
        reinit=True
    )
    try:
        trial_dir = session.get_trial_dir()
        output_dir = os.path.join(trial_dir, "results")
    except Exception as e:
        logger.error(f"path error: {str(e)}")
        raise

    training_args = TrainingArguments(
        run_name=run.name,
        report_to="wandb", 
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        learning_rate=config["learning_rate"],
        weight_decay=0.01,
        logging_dir=os.path.join(trial_dir, "logs"),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,
        metric_for_best_model="eval_accuracy",
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
        logger.error(f"train fail: {str(e)}")
        raise

    try:
        eval_results = trainer.evaluate()
    except Exception as e:
        logger.error(f"eval fail: {str(e)}")
        raise

    try:
        trainer.save_model(output_dir)
        tune.report(
            metrics=eval_results,
            checkpoint=tune.Checkpoint.from_directory(output_dir)
        )
        wandb.log({
            "trial_learning_rate": config["learning_rate"],
            "trial_accuracy": eval_results["eval_accuracy"],
            "trial_f1": eval_results["eval_f1"],
        })
    except Exception as e:
        logger.error(f"log fail: {str(e)}")
        raise

def get_next_model_version(save_path = "models/xln_pytorch/", model_prefix="XLN-v"):
    if not os.path.exists(save_path):
        return 1

    version_pattern = re.compile(rf"^{re.escape(model_prefix)}(\d+)$")
    max_version = 0

    for name in os.listdir(save_path):
        match = version_pattern.match(name)
        if match:
            version_num = int(match.group(1))
            if version_num > max_version:
                max_version = version_num

    return max_version + 1

def export_model_to_onnx(model, tokenizer, output_path, model_name):
    if os.path.isdir(output_path):
        output_file = os.path.join(output_path, f"{model_name}.onnx")
    else:
        output_file = output_path
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    model.eval()
    try:
        sample_text = "This is a sample text for ONNX export."
        encoded_input = tokenizer(
            sample_text,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super(ModelWrapper, self).__init__()
                self.model = model
            def forward(self, input_ids, attention_mask, token_type_ids=None):
                if token_type_ids is not None:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids
                    )
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                return outputs.logits
        wrapped_model = ModelWrapper(model)
        input_names = ["input_ids", "attention_mask"]
        if "token_type_ids" in encoded_input:
            input_names.append("token_type_ids")
        dynamic_axes = {
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'output': {0: 'batch_size', 1: 'num_classes'}
        }
        if "token_type_ids" in encoded_input:
            dynamic_axes["token_type_ids"] = {0: 'batch_size', 1: 'sequence_length'}
        inputs_dict = {
            'input_ids': encoded_input['input_ids'],
            'attention_mask': encoded_input['attention_mask']
        }
        if "token_type_ids" in encoded_input:
            inputs_dict["token_type_ids"] = encoded_input["token_type_ids"]
        with torch.no_grad():
            torch.onnx.export(
                wrapped_model,
                tuple(inputs_dict.values()),
                output_file,
                input_names=input_names,
                output_names=["output"],
                dynamic_axes=dynamic_axes,
                opset_version=12,
                do_constant_folding=True,
                export_params=True,
                verbose=False
            )
        if os.path.exists(output_file):
            logger.info(f"Model successfully exported to: {output_file}")
            return output_file
        else:
            logger.warning(f"Export failed: File was not saved at {output_file}")
            return None
    except Exception as e:
        logger.error(f"ONNX export failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

def export_with_transformers_api(model, tokenizer, output_file):
    try:
        model.eval()
        feature = "sequence-classification"
        model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature=feature)
        onnx_config = model_onnx_config(model.config)
        dummy_inputs = tokenizer("This is a test", return_tensors="pt")
        export(
            preprocessor=tokenizer,
            model=model,
            config=onnx_config,
            opset=12,
            output=Path(output_file),
            tokenizer=tokenizer,
            feature=feature
        )
        return output_file
    except Exception as e:
        logger.error(f"Transformers API export failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_offline():
    retcode = pytest.main([
        "tests",
        "--disable-warnings",
        "-v",
        "-s"
    ])
    return retcode

def fakenews_run(WANDB_KEY):
    # wandb.login(key=WANDB_KEY)
    wandb.login(
        key  = WANDB_KEY,
        host = os.environ["WANDB_HOST"],
    )
    
    df = pd.read_csv("../etl_data/task2_data/welfake_train.csv")
    
    df = df.dropna()
    df['text'] = df['title'] + " " + df['text']
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, eval_df = train_test_split(train_df, test_size=0.2, random_state=42)

    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', cache_dir='../models/xln_source')
    train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
    eval_dataset = Dataset.from_pandas(eval_df[['text', 'label']])
    test_dataset = Dataset.from_pandas(test_df[['text', 'label']])

    tokenize_fn = partial(preprocess, tokenizer=tokenizer)
    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    eval_dataset = eval_dataset.map(tokenize_fn, batched=True)
    test_dataset = test_dataset.map(tokenize_fn, batched=True)

    model = XLNetForSequenceClassification.from_pretrained(
        'xlnet/xlnet-base-cased',
        num_labels=2,
        problem_type="single_label_classification",
        cache_dir='../models/xln_source'
    )
    search_space = {
        "learning_rate": tune.grid_search([1e-5]),
    }
    model_id = get_next_model_version()
    save_path = f"models/XLN/{model_id}"
    save_path.mkdir(parents=True, exist_ok=True)
    model_name = f"XLN-v{model_id}"
    torch_path = f"models/xln_pytorch/{model_name}"

    run_name = f"{model_name}_{datetime.now().strftime('%m%d_%H%M')}"

    current_dir = os.getcwd()
    storage_path = f"file://{current_dir}/ray_results/fakenews_results"

    train_fn_with_params = tune.with_parameters(train_fn, model=model, train_dataset=train_dataset, eval_dataset=eval_dataset, run_name = run_name)
    ray.init(_temp_dir=f"./ray_tmp", ignore_reinit_error=True)

    analysis = tune.run(
        train_fn_with_params,
        config=search_space,
        resources_per_trial={"cpu": 0, "gpu": 1},
        num_samples=1,
        verbose=1,
        storage_path=storage_path,
        callbacks=[],
    )
    best_trial = analysis.get_best_trial(metric="eval_accuracy", mode="max")
    best_checkpoint = best_trial.checkpoint
    best_checkpoint_dir = best_checkpoint.to_directory()

    best_model = XLNetForSequenceClassification.from_pretrained(best_checkpoint_dir)
    best_model.save_pretrained("/app/models/tmp/latest_model")
    torch.save(test_dataset, "/app/models/tmp/test_dataset.pt")
    retcode = evaluate_offline()

    onnx_path = "fail"

    if retcode != 0:
        logger.warning("test failed")
    else:
        best_model.save_pretrained(torch_path)
        onnx_path = export_model_to_onnx(best_model, tokenizer, save_path, model_name)
        logger.info(f"New model exported: {model_name}, path: {onnx_path}")
        logger.info(model_name)
    wandb.finish()
    return onnx_path, model_name

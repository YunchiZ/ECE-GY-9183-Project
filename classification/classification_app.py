import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import wandb
import torch
from sklearn.metrics import accuracy_score, f1_score
import os
os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
import ray
from ray import tune
from ray.air import session
from pathlib import Path
import json
from filelock import FileLock
import pytest
import subprocess
from dataset_classes import NewsDataset
from datetime import datetime
import pickle

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




def train_fn(config, model, train_dataset, eval_dataset, run_name):
    trial_id = session.get_trial_name()
    wandb.init(
        project="Mlops-classification",
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
    # base_path = Path(base_dir)
    # base_path.mkdir(exist_ok=True)  # 创建 model 文件夹（如果不存在）
    
    # train_dir = Path(__file__).resolve().parent.parent
    # model_dir = train_dir / "model"
    # model_dir.mkdir(parents=True, exist_ok=True)
    # base_path = model_dir
    
    # existing_versions = [
    #     int(p.name.replace("BERT-v", ""))
    #     for p in base_path.iterdir()
    #     if p.is_dir() and p.name.startswith("BERT-v") and p.name.replace("BERT-v", "").isdigit()
    # ]
    # next_version = max(existing_versions + [-1]) + 1  # 如果不存在则为 v0
    task_id = "2"

    current_dir = Path(__file__).resolve().parent
    parent_dir = current_dir.parent
    status_path = parent_dir / "model/model_status.json"
    
    lock_path = status_path.with_suffix(".lock")
    lock = FileLock(str(lock_path))
    with lock:
        if not status_path.exists():
            return None  # 没有记录

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
            if model_name.startswith("BERT-v"):
                try:
                    version_num = int(model_name.replace("BERT-v", ""))
                    versions.append(version_num)
                except ValueError:
                    continue

        return max(versions) if versions else -1
    # return f"BERT-v{next_version}", base_path / f"BERT-v{next_version}"


def update_model_status(new_model_name):
    task_id = "2"
    
    current_dir = Path(__file__).resolve().parent
    parent_dir = current_dir.parent
    status_path = parent_dir / "model/model_status.json"
    
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

def export_model_to_onnx(model, tokenizer, output_path, model_name):
    """
    Export a model to ONNX format with proper error handling.
    
    Args:
        model: The HuggingFace model to export
        tokenizer: The tokenizer for the model
        output_path: The path to save the ONNX model
        
    Returns:
        The path to the saved ONNX file
    """
    import os
    from pathlib import Path
    import torch
    
    # Ensure output_path is a file path, not just a directory
    if os.path.isdir(output_path):
        output_file = os.path.join(output_path, f"{model_name}.onnx")
    else:
        output_file = output_path
        
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"Exporting model to: {output_file}")
    
    # Set the model to evaluation mode
    model.eval()
    
    try:
        # Create sample inputs specifically for XLNet sequence classification
        sample_text = "This is a sample text for ONNX export."
        encoded_input = tokenizer(
            sample_text,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        
        # Fix for dimension mismatch: Create a custom forward wrapper
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super(ModelWrapper, self).__init__()
                self.model = model
                
            def forward(self, input_ids, attention_mask, token_type_ids=None):
                # Process inputs to ensure dimensions match
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
        
        # Wrap the model
        wrapped_model = ModelWrapper(model)
        
        # Get input names
        input_names = ["input_ids", "attention_mask"]
        if "token_type_ids" in encoded_input:
            input_names.append("token_type_ids")
        
        # Define dynamic axes for variable input sizes
        dynamic_axes = {
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'output': {0: 'batch_size', 1: 'num_classes'}
        }
        
        if "token_type_ids" in encoded_input:
            dynamic_axes["token_type_ids"] = {0: 'batch_size', 1: 'sequence_length'}
        
        # Create inputs dictionary
        inputs_dict = {
            'input_ids': encoded_input['input_ids'],
            'attention_mask': encoded_input['attention_mask']
        }
        
        if "token_type_ids" in encoded_input:
            inputs_dict["token_type_ids"] = encoded_input["token_type_ids"]
        
        # Export directly using torch.onnx - UPDATED OPSET VERSION FROM 12 TO 14
        with torch.no_grad():
            torch.onnx.export(
                wrapped_model,
                tuple(inputs_dict.values()),  # Input to the model
                output_file,
                input_names=input_names,
                output_names=["output"],
                dynamic_axes=dynamic_axes,
                opset_version=14,  # Changed from 12 to 14 to support scaled_dot_product_attention
                do_constant_folding=True,
                export_params=True,
                verbose=False  # Set to True for more diagnostic info
            )
        
        # Verify the model was saved
        if os.path.exists(output_file):
            print(f"✅ Model successfully exported to: {output_file}")
            return output_file
        else:
            print(f"❌ Export failed: File was not saved at {output_file}")
            return None
            
    except Exception as e:
        print(f"❌ ONNX export failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


# Alternative export method using the transformers API
def export_with_transformers_api(model, tokenizer, output_file):
    """
    Try exporting with the transformers ONNX API
    Only use this as a fallback if the direct torch method fails
    """
    try:
        from transformers.onnx import export
        from transformers.onnx.features import FeaturesManager
        from pathlib import Path
        
        # Set model to evaluation mode
        model.eval()
        
        # For XLNet sequence classification
        feature = "sequence-classification"
        model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature=feature)
        onnx_config = model_onnx_config(model.config)
        
        # Create dummy inputs
        dummy_inputs = tokenizer("This is a test", return_tensors="pt")
        
        # Export - already using opset 14 here
        export(
            preprocessor=tokenizer,
            model=model,
            config=onnx_config,
            opset=14,
            output=Path(output_file),
            tokenizer=tokenizer,
            feature=feature
        )
        
        return output_file
    except Exception as e:
        print(f"Transformers API export failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':

    data = pd.read_csv("./dataset/NewsCategorizer.csv")
    
    train_texts, test_texts, train_labels, test_labels = train_test_split(data['short_description'], data['category'], test_size=0.2, shuffle=True)
    train_texts, eval_texts, train_labels, eval_labels = train_test_split(data['short_description'], data['category'], test_size=0.2, shuffle=True)

    # label_encoder = LabelEncoder()
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    
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

    # wandb.init(project="Mlops-classification", entity="yunchiz-new-york-university")

    # model_name, save_path = get_next_model_version()

    model_id = get_next_model_version() + 1
    
    train_dir = Path(__file__).resolve().parent.parent
    save_path = train_dir / f"model/BERT/{model_id}"
    save_path.mkdir(parents=True, exist_ok=True)
    model_name = f"BERT-v{model_id}"
    
    run_name = f"{model_name}_{datetime.now().strftime('%m%d_%H%M')}"
    os.environ["WANDB_PROJECT"] = "Mlops-classification"
    os.environ["WANDB_DISABLED"] = "false"
    
    current_dir = os.getcwd()
    storage_path = f"file://{current_dir}/ray_results/classification_results"
    
    
    train_fn_with_params = tune.with_parameters(train_fn, model=model, train_dataset=train_dataset, eval_dataset=eval_dataset, run_name = run_name)
    ray.init(ignore_reinit_error=True)  # Initialize Ray
    analysis = tune.run(
        train_fn_with_params,  # The training function that Ray Tune will use
        config=search_space,  # The search space of hyperparameters
        resources_per_trial={"cpu": 0, "gpu": 1},
        num_samples=1,  # Number of trials (hyperparameter combinations)
        verbose=1,  # Verbosity level of Ray Tune
        storage_path=storage_path,
        callbacks=[],
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
    
    retcode = evaluate_offline()
    if retcode != 0:
        print("test failed")
    else:
        onnx_path = export_model_to_onnx(best_model, tokenizer, save_path, model_name)
        print(f"New model exported: {model_name}, path: {onnx_path}")
        update_model_status(model_name)
        print(model_name)
    wandb.finish()



    
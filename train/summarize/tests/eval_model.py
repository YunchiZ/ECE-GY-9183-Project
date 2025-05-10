from transformers import BartTokenizer, BartForConditionalGeneration, TrainingArguments, Trainer
import numpy as np
import evaluate
import torch

gen_kwargs = {
    "max_length": 128,
    "min_length": 30,
    "num_beams": 4,
    "length_penalty": 2.0,
    "no_repeat_ngram_size": 3,
    "early_stopping": True,
}

def evaluate_model_rouge(test_data, model, tokenizer):
    try:
        # move model to gpu
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        pred_summaries = []
        true_summaries = []
        
        batch_size = 8
        for i in range(0, len(test_data), batch_size):
            batch = test_data.select(range(i, min(i+batch_size, len(test_data))))
            
            inputs = tokenizer(
                batch["article"],
                max_length=1024,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                summaries = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **gen_kwargs
                )
            
            decoded_preds = tokenizer.batch_decode(summaries, skip_special_tokens=True)
            decoded_labels = [highlight for highlight in batch["highlights"]]
            
            pred_summaries.extend(decoded_preds)
            true_summaries.extend(decoded_labels)
            
            print(f"Processed {i + batch_size}/{len(test_data)} samples")
            
        # rouge
        rouge = evaluate.load("rouge")
        results = rouge.compute(
            predictions=pred_summaries,
            references=true_summaries,
            use_stemmer=True
        )
        
        return {
            "rouge1": round(results["rouge1"], 4),
            "rouge2": round(results["rouge2"], 4),
            "rougeL": round(results["rougeL"], 4),
        }
    
    except Exception as e:
        print(f"评估过程中发生错误: {str(e)}")
        raise


    


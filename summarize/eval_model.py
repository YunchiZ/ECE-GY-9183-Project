from transformers import BartTokenizer, BartForConditionalGeneration, TrainingArguments, Trainer
import numpy as np
import evaluate
import torch

gen_kwargs = {
    "max_length": 128,          # 生成摘要的最大长度
    "min_length": 30,           # 生成摘要的最小长度
    "num_beams": 4,             # Beam Search 的 beam 数
    "length_penalty": 2.0,      # 长度惩罚系数（>1鼓励更长，<1鼓励更短）
    "no_repeat_ngram_size": 3,  # 禁止重复的 n-gram 大小
    "early_stopping": True,     # 是否提前停止生成
}

def evaluate_model_rouge(test_data, model, tokenizer):
    """
    评估测试集并返回 ROUGE 指标
    Args:
        test_data (Dataset): 预处理后的测试数据集（需包含 "article" 和 "highlights"）
        model (PreTrainedModel): 加载的最佳模型
        tokenizer (PreTrainedTokenizer): 对应的 tokenizer
    Returns:
        dict: ROUGE 指标结果
    """
    try:
        # 将模型移动到 GPU（如果可用）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # 存储预测和真实摘要
        pred_summaries = []
        true_summaries = []
        
        # 批量生成预测
        batch_size = 8  # 根据 GPU 显存调整
        for i in range(0, len(test_data), batch_size):
            batch = test_data.select(range(i, min(i+batch_size, len(test_data))))
            
            # 编码输入文本
            inputs = tokenizer(
                batch["article"],
                max_length=1024,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            ).to(device)
            
            # 生成摘要
            with torch.no_grad():
                summaries = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **gen_kwargs
                )
            
            # 解码预测和真实摘要
            decoded_preds = tokenizer.batch_decode(summaries, skip_special_tokens=True)
            decoded_labels = [highlight for highlight in batch["highlights"]]
            
            pred_summaries.extend(decoded_preds)
            true_summaries.extend(decoded_labels)
            
            print(f"Processed {i + batch_size}/{len(test_data)} samples")
            
        # 计算 ROUGE 指标
        rouge = evaluate.load("rouge")
        results = rouge.compute(
            predictions=pred_summaries,
            references=true_summaries,
            use_stemmer=True
        )
        
        # 格式化结果（保留4位小数）
        return {
            "rouge1": round(results["rouge1"], 4),
            "rouge2": round(results["rouge2"], 4),
            "rougeL": round(results["rougeL"], 4),
        }
    
    except Exception as e:
        print(f"评估过程中发生错误: {str(e)}")
        raise


    


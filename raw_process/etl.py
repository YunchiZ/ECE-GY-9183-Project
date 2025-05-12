import os
from extract import load_csv, load_csv_welfake, load_csv_summary
from filter import clean_text, clean_text_welfake, clean_text_summary
from trans import normalize_text, normalize_text_welfake, normalize_text_summary
from load import save_jsonl
from split import split_data

def run_etl_classification(output_dir):

    df = load_csv()
    df = clean_text(df)
    df = normalize_text(df)
    train, val, late = split_data(df)

    os.makedirs(output_dir, exist_ok=True)
    save_jsonl(train, f"{output_dir}/train.jsonl")
    save_jsonl(val, f"{output_dir}/val.jsonl")
    save_jsonl(late, f"{output_dir}/late_data.jsonl")

def run_etl_welfake(output_dir):
    df = load_csv_welfake()
    df = clean_text_welfake(df)
    df = normalize_text_welfake(df)
    train, val, late = split_data(df)
    
    os.makedirs(output_dir, exist_ok=True)
    save_jsonl(train, f"{output_dir}/train.jsonl")
    save_jsonl(val, f"{output_dir}/val.jsonl")
    save_jsonl(late, f"{output_dir}/late_data.jsonl")
    
def run_etl_summary(output_dir):
    df = load_csv_summary()
    df = clean_text_summary(df)
    df = normalize_text_summary(df)
    train, val, late = split_data(df)
    
    os.makedirs(output_dir, exist_ok=True)
    save_jsonl(train, f"{output_dir}/train.jsonl")
    save_jsonl(val, f"{output_dir}/val.jsonl")
    save_jsonl(late, f"{output_dir}/late_data.jsonl")

if __name__ == "__main__":

    run_etl_classification("output/classification")
    run_etl_welfake("output/welfake")
    run_etl_summary("output/summary")
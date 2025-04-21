import os
import logging
from flask import Flask, request

from extract import load_csv, load_csv_welfake, load_csv_summary
from filter import clean_text, clean_text_welfake, clean_text_summary
from trans import normalize_text, normalize_text_welfake, normalize_text_summary
from load import save_jsonl
from split import split_data

etl_data_dir = '/app/etl_data'
log_file = os.path.join(etl_data_dir, "app.log")

os.makedirs(etl_data_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)

app = Flask(__name__)

def run_etl(df_loader, cleaner, normalizer, output_subdir):
    df = df_loader()
    df = cleaner(df)
    df = normalizer(df)

    logging.info(f"First few rows:\n{df.head()}")
    train, val, late = split_data(df)

    output_dir = os.path.join(etl_data_dir, output_subdir)
    os.makedirs(output_dir, exist_ok=True)
    save_jsonl(train, f"{output_dir}/train.jsonl")
    save_jsonl(val, f"{output_dir}/val.jsonl")
    save_jsonl(late, f"{output_dir}/late_data.jsonl")

    logging.info(f"ETL for {output_subdir} complete. Output saved in {output_dir}")
    return True


@app.route("/etl", methods=["POST"])
def trigger_etl():
    data = request.get_json(force=True)
    task = data.get("task")

    if not task:
        return {"error": "Missing task field"}, 400

    try:
        if task == "classification":
            run_etl(load_csv, clean_text, normalize_text, "classification")
        elif task == "welfake":
            run_etl(load_csv_welfake, clean_text_welfake, normalize_text_welfake, "welfake")
        elif task == "summary":
            run_etl(load_csv_summary, clean_text_summary, normalize_text_summary, "summary")
        else:
            return {"error": f"Unknown task: {task}"}, 400
    except Exception as e:
        logging.exception("ETL failed")
        return {"error": str(e)}, 500

    return {"status": "ETL success", "task": task}, 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

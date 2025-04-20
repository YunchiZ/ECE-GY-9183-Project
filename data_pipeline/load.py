def save_jsonl(df, path):
    df.to_json(path, orient="records", lines=True)

import pandas as pd

## change the path to read from local environment

def load_csv():
    df = pd.read_csv("NewsCategorizer.csv")
    df.columns = df.columns.str.strip()
    return df

def load_csv_welfake():
    df = pd.read_csv("WELFake_Dataset.csv", encoding='utf-8')
    df.columns = df.columns.str.strip()
    return df

def load_csv_summary():
    df = pd.read_csv("summarization.csv", encoding='utf-8')
    df.columns = df.columns.str.strip()
    return df

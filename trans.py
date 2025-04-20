import re

def normalize_text(df):
    df["text"] = df["headline"].str.strip() + " " + df["short_description"].str.strip()
    df = df[["category", "text"]].reset_index(drop=True)
    return df

def normalize_text_welfake(df):
    df["text"] = df["title"].str.strip() + " " + df["text"].str.strip()
    df = df[["label", "text"]].reset_index(drop=True)
    return df

def normalize_text_summary(df):

    df["document"] = df["document"].astype(str).apply(clean_document_text)
    df["summary"] = df["summary"].astype(str).apply(clean_document_text)
    df["summary"] = df["summary"].astype(str).apply(lambda x: " ".join(x.split()))
    return df

def clean_document_text(text):
    if not isinstance(text, str):
        return ""

    text = re.sub(r'^.*?--\s*', '', text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    cleaned_lines = []
    for line in lines:
        if not re.search(r'[.?!…]("|”)?$', line):
            line += "."
        cleaned_lines.append(line)
    text = " ".join(cleaned_lines)
    text = re.sub(r'\s+([.?!,:;])', r'\1', text)
    text = re.sub(r'\.(\S)', r'. \1', text)

    return text.strip()

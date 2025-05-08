import re, html, unicodedata
from transformers import AutoTokenizer
import numpy as np

# -------- Data Filtering --------
# This section includes:
# 1) HTML entity decoding & full-width → half-width conversion
# 2) Remove control characters (clean hidden symbols)
# 3) Append punctuation marks to lines to ensure proper ending
# 4) Add spaces between sentences
# 5) Normalize multiple spaces
# 6) Remove empty lines and strip whitespace
# 7) Validate length via Pydantic
# Pre-compilation
MULTISPACE_RX   = re.compile(r"\s+")
DOT_NOGAP_RX    = re.compile(r"\.(\S)")
NO_PUNCT_RX     = re.compile(r'[.?!…]("|”)?$')  # Ensure punctuation at the end of lines
CTRL_CHAR_RX  = re.compile(r"[\x00-\x1f\x7f]")  # ASCII control characters

def data_filter(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # Basic cleaning
    text = unicodedata.normalize("NFKC", html.unescape(text.strip()))
    text = CTRL_CHAR_RX.sub("", text)          # Remove control characters

    # Append punctuation marks to lines
    lines = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if not NO_PUNCT_RX.search(ln):
            ln += "."
        lines.append(ln)

    text = " ".join(lines)
    text = MULTISPACE_RX.sub(" ", text)
    text = DOT_NOGAP_RX.sub(r". \1", text)

    return text.strip()


# Tokenization

TOKENS = {
    "BART": AutoTokenizer.from_pretrained("facebook/bart-base", cache_dir="./models"),
    "XLN" : AutoTokenizer.from_pretrained("xlnet-base-cased", cache_dir="./models"),
    "BERT": AutoTokenizer.from_pretrained("distilbert-base-uncased", cache_dir="./models")
}

MAXLEN = {
    "BART": 1024,
    "XLN" : 256,
    "BERT": 512
}  # However, the frontend limits word count to generally less than 1200 characters (about 120 words),
   # so this MAXLEN is essentially unused.

def build_payloads(raw_text: str) -> dict:
    """
    Package the raw text into payload dictionaries required for Triton inference for three models.
    """
    payloads = {}
    for model in ("BART", "XLN", "BERT"):
        token = TOKENS[model](raw_text, padding="max_length",
                              truncation=True, max_length=MAXLEN[model],
                              return_tensors="np")

        input_ids, attention_mask = token["input_ids"], token["attention_mask"]

        payloads[model] = {
            "inputs": [
                {
                    "name": "input_ids",
                    "shape": list(input_ids.shape),  # [1, L]
                    "datatype": "INT64",
                    "data": input_ids.flatten().tolist()
                },
                {
                    "name": "attention_mask",
                    "shape": list(attention_mask.shape),
                    "datatype": "INT64",
                    "data": attention_mask.flatten().tolist()
                }
            ],
            "outputs": (
                [{"name": "output_ids"}] if model == "BART"
                else [{"name": "logits"}]
            )
        }

    return payloads
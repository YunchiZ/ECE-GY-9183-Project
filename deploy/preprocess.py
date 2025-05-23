import re, html, unicodedata
from transformers import AutoTokenizer
import numpy as np
import logging
import os

train_data_dir = "/app/models"
deploy_data_dir = "/app/deploy_data"

# ---------------- Logging Initialization ----------------
log_file = os.path.join(train_data_dir, "app.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a",
)

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
    "BART": AutoTokenizer.from_pretrained("facebook/bart-base", cache_dir="/app/tokenizer/bart_source"),
    "XLN" : AutoTokenizer.from_pretrained("xlnet-base-cased", cache_dir="/app/tokenizer/xln_source"),
    "BERT": AutoTokenizer.from_pretrained("distilbert-base-uncased", cache_dir="/app/tokenizer/bert_source")
}

MAXLEN = {
    "BART": 1024,
    "XLN" : 256,
    "BERT": 512
}  # However, the frontend limits word count to generally less than 1200 characters (about 120 words),
   # so this MAXLEN is essentially unused.


def build_payloads(raw_text: str) -> dict:
    import numpy as np
    payloads = {}

    for model in ("BART", "XLN", "BERT"):
        token = TOKENS[model](
            raw_text,
            padding=True,
            truncation=True,
            max_length=MAXLEN[model],
            return_tensors="np"
        )

        input_ids = token["input_ids"]
        attention_mask = token["attention_mask"]

        if model == "BART":
            # input_ids 和 attention_mask: shape 为 [seq_len]
            input_ids_1d = input_ids.squeeze(0)  # 从 [1, L] → [L]
            attention_mask_1d = attention_mask.squeeze(0)

            payloads["BART"] = {
                "inputs": [
                    {
                        "name": "input_ids",
                        "shape": [input_ids_1d.shape[0]],  # e.g. [476]
                        "datatype": "INT64",
                        "data": input_ids_1d.tolist()
                    },
                    {
                        "name": "attention_mask",
                        "shape": [attention_mask_1d.shape[0]],
                        "datatype": "INT64",
                        "data": attention_mask_1d.tolist()
                    }
                ],
                "outputs": [
                    {"name": "logits"}
                ]
            }

            logging.info("input_ids shape (1D): %s", input_ids_1d.shape)
            logging.info("attention_mask shape (1D): %s", attention_mask_1d.shape)

            logging.info("input_ids shape: %s", input_ids.shape)
            logging.info("attention_mask shape: %s", attention_mask.shape)

        elif model == "XLN":
            token_type_ids = token.get("token_type_ids", np.zeros_like(input_ids, dtype=np.int64))
            payloads["XLN"] = {
                "inputs": [
                    {
                        "name": "input_ids",
                        "shape": list(input_ids.shape),
                        "datatype": "INT64",
                        "data": input_ids.flatten().tolist()
                    },
                    {
                        "name": "attention_mask",
                        "shape": list(attention_mask.shape),
                        "datatype": "INT64",
                        "data": attention_mask.flatten().tolist()
                    },
                    {
                        "name": "token_type_ids",
                        "shape": list(token_type_ids.shape),
                        "datatype": "INT64",
                        "data": token_type_ids.flatten().tolist()
                    }
                ],
                "outputs": [{"name": "output"}]
            }

        elif model == "BERT":
            payloads["BERT"] = {
                "inputs": [
                    {
                        "name": "input_ids",
                        "shape": list(input_ids.shape),
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
                "outputs": [{"name": "output"}]
            }

    return payloads

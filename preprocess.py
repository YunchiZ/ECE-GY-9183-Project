import re, html, unicodedata
from transformers import AutoTokenizer
import numpy as np

# -------- Data Filtering --------
# 该部分内容:
# 1) HTML 实体反转 & 全角→半角
# 2) 去控制字符 (隐藏符号清理)
# 3) 行级补标点 保证句末有标点
# 4) 句间补空格
# 5) 多空白归一
# 6) 去空行 strip
# 7) pydantic 长度验证
# 预编译
MULTISPACE_RX   = re.compile(r"\s+")
DOT_NOGAP_RX    = re.compile(r"\.(\S)")
NO_PUNCT_RX     = re.compile(r'[.?!…]("|”)?$')  # noinspection
CTRL_CHAR_RX  = re.compile(r"[\x00-\x1f\x7f]")  # ASCII 控制符

def data_filter(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # 基础清洗
    text = unicodedata.normalize("NFKC", html.unescape(text.strip()))
    text = CTRL_CHAR_RX.sub("", text)          # 去控制字符

    # 行级补标点
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
}  # 但是前端会限制字数一般来说小于1200个字符 也就是120个单词 所以这个MAXLEN形同虚设

def build_payloads(raw_text: str) -> dict:
    """
    将原始文本封装成三个模型Triton推理所需要的payload字典
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

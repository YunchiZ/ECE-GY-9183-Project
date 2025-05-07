import os, re, json, shutil, logging, httpx, asyncio, tempfile
from filelock import FileLock, Timeout as LockTimeout
from json import JSONDecodeError
from typing import Any, Dict, Literal
import random


MODEL_STATUS_FILE = "/app/models/model_status.json"
TRAIN_DATA_DIR = "/app/models"


async def notify_docker(docker: str, route: str, payload: Dict[str, Any],
    *,
    retries: int = 3,          # 最多重试次数（含首次）
    backoff: float = 1.5,      # 指数退避倍率
    base_delay: float = 0.5    # 首次失败后的等待秒数
) -> bool:
    """
    可靠地向 `${docker}:8000/{route}` 发送 POST。
    成功返回 True，全部重试仍失败则 False。
    1) 连接 + 读取双超时 5 s；
    2) 5xx 或网络错误都会重试；
    3) 指数退避 + ±10% 抖动。
    """
    url = f"http://{docker}:8000/{route}"
    timeout = httpx.Timeout(5.0, read=5.0)      # 连接、读取各 5 s
    for attempt in range(1, retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(url, json=payload)
                if r.is_success:                # 2xx
                    logging.info("[notify] %s %s → %s (try %d)",docker, payload, r.status_code, attempt)
                    return True
                # 4xx: 语法错误/鉴权问题，一般没必要重试
                if 400 <= r.status_code < 500:
                    logging.error("[notify] client error %s: %s", r.status_code, r.text)
                    return False
                # 5xx 走到下面统一重试
                raise httpx.HTTPStatusError(f"Server error {r.status_code}", request=r.request, response=r)

        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            logging.warning("[notify] attempt %d/%d failed: %s", attempt, retries, e)

            if attempt == retries:
                break

            # 指数退避 + 抖动
            delay = base_delay * (backoff ** (attempt - 1))
            delay *= random.uniform(0.9, 1.1)
            await asyncio.sleep(delay)

    logging.error("[notify] GIVE UP after %d tries: %s %s", retries, docker, payload)
    return False


def split_model_tag(tag: str) -> tuple[str, str]:
    """
    将 'BART-v0' → ('BART', '0')
       'BERT-v12' → ('BERT', '12')
    若格式不合法或版本号小于 0，会记录错误日志并抛出 ValueError。
    """
    try:
        # 从右边切分一次，获取模型名称和版本号
        name, v = tag.rsplit("-v", 1)
        if not name or not v.isdigit():
            raise ValueError(f"Invalid tag format: '{tag}', expected <name>-v<digit+>")

        # 转换版本号为整数
        version = int(v)  # 转为整数以验证其合法性
        if version < 0:
            logging.error(f"Invalid version number in tag '{tag}': version must be >= 0")
            raise ValueError(f"Invalid version number in tag '{tag}': version must be >= 0")

        # 返回结果时，版本号转换回字符串
        return name, str(version)

    except ValueError as e:
        logging.error(f"Failed to split model tag '{tag}': {e}")
        raise ValueError(f"Invalid tag format '{tag}': {e}")


def status_modify(cmd: Literal["revoke", "replace"],
                  model: str,
                  index: int) -> None:
    """
    修改 model_status.json，并在需要时删除权重目录。
    • cmd = "revoke"  : 删除 candidate 条目
    • cmd = "replace" : serving→abandon，新 model→serving；abandon >2 时淘汰最老
    """
    lock = FileLock(MODEL_STATUS_FILE + ".lock", timeout=10)

    try:
        with lock:                             # ── 文件级排它
            # 1. 读取 JSON
            try:
                with open(MODEL_STATUS_FILE, "r") as f:
                    status = json.load(f)
            except FileNotFoundError:
                logging.error("Status file not found: %s", MODEL_STATUS_FILE)
                return
            except JSONDecodeError as e:
                logging.error("JSON decode error in %s: %s", MODEL_STATUS_FILE, e)
                return

            key = str(index)
            task_list = status.get(key)
            if not isinstance(task_list, list):
                logging.error("Invalid or missing task[%s] in JSON", key)
                return

            # ---------- cmd 分支 ----------
            if cmd == "revoke":
                entry = next((e for e in task_list
                              if e["model"] == model and e["status"] == "candidate"),
                             None)
                if not entry:
                    logging.warning("No candidate '%s' under task %s", model, key)
                else:
                    task_list.remove(entry)
                    _del_weights(model)
                    logging.info("Revoke candidate %s (task %s)", model, key)

            elif cmd == "replace":
                # 1) 若 abandon ≥2，删除版本号最小的一个
                abandons = [e for e in task_list if e["status"] == "abandon"]
                if len(abandons) >= 2:
                    oldest = min(abandons, key=_version_rank)
                    task_list.remove(oldest)
                    _del_weights(oldest["model"])
                    logging.info("Evict abandon %s (task %s)", oldest["model"], key)

                # 2) serving → abandon
                for e in task_list:
                    if e["status"] == "serving":
                        e["status"] = "abandon"
                        logging.info("Demote serving→abandon %s (task %s)", e["model"], key)
                        break

                # 3) append new serving
                task_list.append({"model": model, "status": "serving"})
                logging.info("Register new serving %s (task %s)", model, key)

            else:
                logging.error("Unknown cmd=%r", cmd)
                return

            # 3. 原子写回
            _atomic_write_json(MODEL_STATUS_FILE, status)
            logging.info("Updated model_status.json for task %s", key)

    except LockTimeout:
        logging.error("Lock %s timeout — skip updating status", MODEL_STATUS_FILE)
    except Exception as e:
        logging.exception("status_modify unexpected error: %s", e)


# ---------- 私有辅助函数 ----------

def _version_rank(entry: dict) -> int:
    """提取模型版本号供排序；解析失败返回极大值。"""
    m = re.match(r".*-v(\d+)$", entry["model"])
    return int(m.group(1)) if m else 1 << 30


def _del_weights(tag: str) -> None:
    """删除权重目录，忽略不存在/权限错误。"""
    try:
        name, ver = tag.rsplit("-v", 1)
        path = os.path.join(TRAIN_DATA_DIR, name, ver)
        shutil.rmtree(path, ignore_errors=True)
    except Exception as e:
        logging.warning("Delete weights of %s failed: %s", tag, e)


def _atomic_write_json(path: str, data) -> None:
    """写入 JSON 到临时文件，再原子替换目标文件。"""
    dir_ = os.path.dirname(path)
    with tempfile.NamedTemporaryFile("w",
                                     dir=dir_,
                                     delete=False,
                                     encoding="utf-8") as tmp:
        json.dump(data, tmp, indent=2, ensure_ascii=False)
        tmp.flush()
        os.fsync(tmp.fileno())              # 落盘
    os.replace(tmp.name, path)              # 原子替换

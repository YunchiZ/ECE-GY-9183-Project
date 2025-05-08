import os, re, json, shutil, logging, httpx, asyncio, tempfile
from filelock import FileLock, Timeout as LockTimeout
from json import JSONDecodeError
from typing import Any, Dict, Literal
import random


MODEL_STATUS_FILE = "/app/models/model_status.json"
TRAIN_DATA_DIR = "/app/models"


async def notify_docker(docker: str, route: str, payload: Dict[str, Any],
    *,
    retries: int = 3,          # Maximum number of retries (including the first attempt)
    backoff: float = 1.5,      # Exponential backoff multiplier
    base_delay: float = 0.5    # Initial delay in seconds after the first failure
) -> bool:
    """
    Reliably send a POST request to `${docker}:8000/{route}`.
    Returns True on success, and False if all retries fail.
    1) Connection + read timeout is set to 5 seconds;
    2) Retries on 5xx or network errors;
    3) Exponential backoff + ±10% jitter.
    """
    url = f"http://{docker}:8000/{route}"
    timeout = httpx.Timeout(5.0, read=5.0)      # 5-second connection and read timeout
    for attempt in range(1, retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(url, json=payload)
                if r.is_success:                # 2xx
                    logging.info("[notify] %s %s → %s (try %d)",docker, payload, r.status_code, attempt)
                    return True
                # 4xx: Client errors like syntax/authentication issues, retries usually unnecessary
                if 400 <= r.status_code < 500:
                    logging.error("[notify] client error %s: %s", r.status_code, r.text)
                    return False
                # 5xx errors will retry below
                raise httpx.HTTPStatusError(f"Server error {r.status_code}", request=r.request, response=r)

        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            logging.warning("[notify] attempt %d/%d failed: %s", attempt, retries, e)

            if attempt == retries:
                break

            # Exponential backoff + jitter
            delay = base_delay * (backoff ** (attempt - 1))
            delay *= random.uniform(0.9, 1.1)
            await asyncio.sleep(delay)

    logging.error("[notify] GIVE UP after %d tries: %s %s", retries, docker, payload)
    return False


def split_model_tag(tag: str) -> tuple[str, str]:
    """
    Split 'BART-v0' → ('BART', '0')
          'BERT-v12' → ('BERT', '12')
    If the format is invalid or the version number is less than 0,
    logs an error and raises a ValueError.
    """
    try:
        # Split from the right once to get the model name and version number
        name, v = tag.rsplit("-v", 1)
        if not name or not v.isdigit():
            raise ValueError(f"Invalid tag format: '{tag}', expected <name>-v<digit+>")

        # Convert version number to integer to validate its legality
        version = int(v)  # Convert to integer to ensure it is valid
        if version < 0:
            logging.error(f"Invalid version number in tag '{tag}': version must be >= 0")
            raise ValueError(f"Invalid version number in tag '{tag}': version must be >= 0")

        # Return the result, converting the version number back to a string
        return name, str(version)

    except ValueError as e:
        logging.error(f"Failed to split model tag '{tag}': {e}")
        raise ValueError(f"Invalid tag format '{tag}': {e}")


def status_modify(cmd: Literal["revoke", "replace"],
                  model: str,
                  index: int) -> None:
    """
    Modify `model_status.json`, and delete weight directories if necessary.
    • cmd = "revoke"  : Remove the candidate entry
    • cmd = "replace" : serving→abandon, new model→serving; if abandon > 2, evict the oldest
    """
    lock = FileLock(MODEL_STATUS_FILE + ".lock", timeout=10)

    try:
        with lock:                             # ── File-level exclusive lock
            # 1. Read JSON
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

            # ---------- Command Branch ----------
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
                # 1) If abandon ≥ 2, delete the one with the smallest version number
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

                # 3) Append new serving
                task_list.append({"model": model, "status": "serving"})
                logging.info("Register new serving %s (task %s)", model, key)

            else:
                logging.error("Unknown cmd=%r", cmd)
                return

            # 3. Write back atomically
            _atomic_write_json(MODEL_STATUS_FILE, status)
            logging.info("Updated model_status.json for task %s", key)

    except LockTimeout:
        logging.error("Lock %s timeout — skip updating status", MODEL_STATUS_FILE)
    except Exception as e:
        logging.exception("status_modify unexpected error: %s", e)


# ---------- Private Helper Functions ----------

def _version_rank(entry: dict) -> int:
    """Extract the model version number for sorting; returns a very large value on failure."""
    m = re.match(r".*-v(\d+)$", entry["model"])
    return int(m.group(1)) if m else 1 << 30


def _del_weights(tag: str) -> None:
    """Delete the weight directory, ignore if it doesn't exist or there are permission errors."""
    try:
        name, ver = tag.rsplit("-v", 1)
        path = os.path.join(TRAIN_DATA_DIR, name, ver)
        shutil.rmtree(path, ignore_errors=True)
    except Exception as e:
        logging.warning("Delete weights of %s failed: %s", tag, e)


def _atomic_write_json(path: str, data) -> None:
    """Write JSON to a temporary file, then atomically replace the target file."""
    dir_ = os.path.dirname(path)
    with tempfile.NamedTemporaryFile("w",
                                     dir=dir_,
                                     delete=False,
                                     encoding="utf-8") as tmp:
        json.dump(data, tmp, indent=2, ensure_ascii=False)
        tmp.flush()
        os.fsync(tmp.fileno())              # Force data to be written to disk
    os.replace(tmp.name, path)              # Atomically replace the target file
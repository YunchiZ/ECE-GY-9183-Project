import os, re, shutil, logging, httpx, asyncio, boto3
from botocore.exceptions import ClientError
from typing import Any, Dict, Literal
import time
import random
from httpx import Response


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
                r: Response = await client.post(url, json=payload)
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


def weight_clean(cmd: Literal["revoke", "replace"], model: str) -> None:
    """
    File-system only clean-up.

    Parameters
    ----------
    cmd   : "revoke"  – delete the weights of *model* immediately
            "replace" – called when a *new* model is promoted to serving:
                        if “two versions earlier” exists, delete it
    model : tag string like "bert-v7"
    """
    if cmd not in {"revoke", "replace"}:
        logging.error("weight_clean: unknown cmd=%r", cmd)
        return

    # ---------------- revoke ----------------
    if cmd == "revoke":
        _del_weights(model)
        logging.info("[revoke] delete weights of %s", model)
        return

    # ---------------- replace ----------------
    # Extract <name> and <ver>
    m = re.match(r"(.+)-v(\d+)$", model)
    if not m:
        logging.info("[replace]Fail in model version extraction: %s", model)
        return

    name, ver_str = m.groups()
    try:
        ver = int(ver_str)
    except ValueError:
        logging.error("[replace] version is not int in tag: %s", model)
        return

    # “two versions earlier”
    old_ver = ver - 2
    if old_ver <= 0:
        logging.info("[replace] nothing to evict for %s (ver<3)", model)
        return

    old_tag  = f"{name}-v{old_ver}"
    old_path = os.path.join(TRAIN_DATA_DIR, name, str(old_ver))

    if os.path.isdir(old_path):
        _del_weights(old_tag)
        logging.info("[replace] evicted old weights %s", old_tag)
    else:
        logging.info("[replace] no 2-versions-old weights to delete (%s)", old_tag)


# --------------------------------------------------------------------------- #
# Helper functions (unchanged from previous version)
# --------------------------------------------------------------------------- #
def _version_rank(entry: dict) -> int:
    """Extract the model version number for sorting; returns big value on failure."""
    m = re.match(r".*-v(\d+)$", entry["model"])
    return int(m.group(1)) if m else 1 << 30


def _del_weights(tag: str) -> None:
    """Delete the weight directory; ignore if it doesn't exist."""
    try:
        name, ver = tag.rsplit("-v", 1)
        path = os.path.join(TRAIN_DATA_DIR, name, ver)
        shutil.rmtree(path, ignore_errors=True)
    except Exception as e:
        logging.warning("Delete weights of %s failed: %s", tag, e)


# =================================================================================
def get_s3():
    """Return an initialized boto3 S3 client (MinIO)."""
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("MINIO_URL"),
        aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("MINIO_SECRET_KEY"),
    )


def ensure_bucket_exists(s3, bucket: str) -> bool:
    """Return True if bucket is accessible, False if not found / not authorised."""
    try:
        s3.head_bucket(Bucket=bucket)
        return True
    except ClientError as e:
        code = e.response["Error"]["Code"]
        # 404 NotFound / 403 Forbidden / 301 MovedPermanently
        logging.error("[MinIO] bucket '%s' inaccessible (%s)", bucket, code)
        return False


def download_candidate_weights(model_tag: str,
                                     bucket: str = "candidate",
                                     max_attempts: int = 3,
                                     base_delay: float = 0.8,
                                     backoff: float = 1.6) -> bool:
    """
    Download <model_tag>.onnx from the given bucket into
      /app/models/<model_name>/<version>/model.onnx

    Retries `max_attempts` times with exponential back-off + 10 % jitter.
    Return True on success, False otherwise.
    """
    try:
        name, ver = split_model_tag(model_tag)
    except ValueError:
        return False

    s3 = get_s3()
    if not ensure_bucket_exists(s3, bucket):
        return False

    # Ensure local directory exists
    local_dir = os.path.join(TRAIN_DATA_DIR, name, ver)
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, "model.onnx")
    key        = f"{model_tag}.onnx"

    for attempt in range(1, max_attempts + 1):
        try:
            s3.download_file(bucket, key, local_path)
            logging.info("[MinIO] downloaded %s → %s (try %d)",
                         key, local_path, attempt)
            return True
        except ClientError as e:
            logging.warning("[MinIO] download attempt %d/%d failed: %s",
                            attempt, max_attempts, e)

            if attempt == max_attempts:
                break

            # back-off + jitter
            delay = base_delay * (backoff ** (attempt - 1))
            delay *= random.uniform(0.9, 1.1)
            time.sleep(delay)

    logging.error("[MinIO] GIVE UP downloading %s from bucket '%s'", key, bucket)
    return False



import httpx, asyncio, logging, subprocess, re, shutil
from typing import Any, Dict, Optional, Union
from func import split_model_tag

# Triton Address
TRITON_HOST     = "http://triton:8000"
INFER_URL_TMPL  = TRITON_HOST + "/v2/models/{model}/versions/{ver}/infer"
REPO_URL_TMPL   = TRITON_HOST + "/v2/repository/models/{model}/{action}"
READY_URL_TMPL  = TRITON_HOST + "/v2/models/{model}{ver_part}/ready"

DEFAULT_INFER_TIMEOUT   = 1.5     # Maximum wait time for a single inference
DEFAULT_MODEL_TIMEOUT   = 120     # Maximum wait time for `wait_until_ready`


def _mk_timeout(read: float) -> httpx.Timeout:
    return httpx.Timeout(connect=1.0, read=read, write=5.0, pool=5.0)


async def triton_infer(model: str,
                       version: int,
                       payload: Dict[str, Any],
                       timeout: float = DEFAULT_INFER_TIMEOUT) -> Optional[dict]:
    """
    Send an inference request; returns Triton JSON on success, None on failure.
    """
    url = INFER_URL_TMPL.format(model=model, ver=version)
    try:
        async with httpx.AsyncClient(timeout=_mk_timeout(timeout)) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            return r.json()

    except httpx.ReadTimeout:
        logging.warning("[Triton] %s:%s infer read-timeout %.1fs", model, version, timeout)
    except httpx.HTTPStatusError as e:
        logging.error("[Triton] %s:%s HTTP %s – %s",
                      model, version, e.response.status_code, e.response.text[:200])
    except httpx.RequestError as e:
        logging.error("[Triton] %s:%s request error: %s", model, version, e)
    except Exception as e:
        logging.exception("[Triton] %s:%s unexpected error: %s", model, version, e)
    return None


# Simple validation of information passed during API calls: 1000 words * 12 characters/word = max 12000 characters
# Standard Triton response format:
'''
{
  "model_name": "bert_v1",
  "outputs": [
    {
      "name": "prob",
      "datatype": "FP32",
      "shape": [1,2],
      "data": [0.12, 0.88]
    }
  ]
}
'''

VersionT = Union[str, int]
async def triton_config(model_name: str,
                        version: Optional[VersionT] = None,
                        action: str = "load",
                        wait_until_ready: bool = True,
                        timeout: int = DEFAULT_MODEL_TIMEOUT) -> None:
    """
    action = "load" / "unload"
    Wait for the model to be READY for up to *timeout* seconds.
    """
    if action not in ("load", "unload"):
        raise ValueError("action must be 'load' or 'unload'")

    body: Dict[str, Any] = {}
    if version is None:
        ver_str: Optional[str] = None
    elif isinstance(version, int):
        ver_str = str(version)
    else:  # str
        ver_str = version.strip() or None  # Empty string is treated as None
    if ver_str is not None:
        body["parameters"] = {"version": ver_str}

    timeout_obj = _mk_timeout(read=10.0)
    async with httpx.AsyncClient(timeout=timeout_obj) as client:
        # ① Send load/unload command
        url = REPO_URL_TMPL.format(model=model_name, action=action)
        r = await client.post(url, json=body)
        r.raise_for_status()
        logging.info("[Triton] %s %s:%s accepted",
                     action, model_name, ver_str or "<policy>")

        # ② For unload or if there's no need to wait for ready
        if action == "unload" or not wait_until_ready:
            return

        # ③ Poll /ready
        ver_part = f"/versions/{ver_str}" if ver_str is not None else ""
        ready_url = READY_URL_TMPL.format(model=model_name, ver_part=ver_part)

        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            rr = await client.get(ready_url)
            if rr.status_code == 200:
                logging.info("[Triton] model %s:%s READY",
                             model_name, ver_str or "<policy>")
                return
            if asyncio.get_event_loop().time() > deadline:
                raise RuntimeError(
                    f"[Triton] wait ready timeout: {model_name}:{ver_str}"
                )
            await asyncio.sleep(0.5)


# Load Test
def _parse_p95(stdout: str) -> float:
    """
    Extract P95 latency (in ms) from perf_analyzer stdout.
    """
    m = re.search(r"95th percentile latency.*?:\s+(\d+)", stdout)
    return int(m.group(1)) / 1e6 if m else 1e9   # ns→ms


def load_test(model_tag: str,
              batch_sizes=(1, 4, 8),
              concurrencies=(1, 4, 8),
              p95_budget_ms: int = 500) -> bool:
    """True = Pass; False = Fail"""
    if shutil.which("perf_analyzer") is None:
        logging.error("perf_analyzer not found in PATH")
        return False

    name, ver = split_model_tag(model_tag)
    for b in batch_sizes:
        for c in concurrencies:
            cmd = [
                "perf_analyzer",
                "-m", name,
                "-v", ver,
                "-b", str(b),
                "--concurrency-range", str(c),
                "--measurement-interval", "3000"
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            p95 = _parse_p95(res.stdout)

            if p95 > p95_budget_ms:
                logging.info("FAIL %s batch=%s concur=%s p95=%.1f ms (budget %d)",
                             model_tag, b, c, p95, p95_budget_ms)
                return False

    logging.info("PASS %s passed load test", model_tag)
    return True
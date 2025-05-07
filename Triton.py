import httpx, asyncio, logging
from typing import Any, Dict, Optional, Union

# Triton Address
TRITON_HOST     = "http://triton:8000"
INFER_URL_TMPL  = TRITON_HOST + "/v2/models/{model}/versions/{ver}/infer"
REPO_URL_TMPL   = TRITON_HOST + "/v2/repository/models/{model}/{action}"
READY_URL_TMPL  = TRITON_HOST + "/v2/models/{model}{ver_part}/ready"

DEFAULT_INFER_TIMEOUT   = 1.5     # 单次推理最长等待
DEFAULT_MODEL_TIMEOUT   = 120     # wait_until_ready 最长等待


def _mk_timeout(read: float) -> httpx.Timeout:
    return httpx.Timeout(connect=1.0, read=read, write=5.0, pool=5.0)


async def triton_infer(model: str,
                       version: int,
                       payload: Dict[str, Any],
                       timeout: float = DEFAULT_INFER_TIMEOUT) -> Optional[dict]:
    """
    发送推理请求；成功返回 Triton JSON，失败返回 None。
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


# API调用时传入信息的简单校验: 1000单词*12字符/单词 = 12000字符最多
# Triton的标准返回形式:
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
    等待模型 READY 最长 *timeout* 秒。
    """
    if action not in ("load", "unload"):
        raise ValueError("action must be 'load' or 'unload'")

    body: Dict[str, Any] = {}
    if version is None:
        ver_str: Optional[str] = None
    elif isinstance(version, int):
        ver_str = str(version)
    else:  # str
        ver_str = version.strip() or None  # 空串视为 None
    if ver_str is not None:
        body["parameters"] = {"version": ver_str}

    timeout_obj = _mk_timeout(read=10.0)
    async with httpx.AsyncClient(timeout=timeout_obj) as client:
        # ① 发送 load / unload 指令
        url = REPO_URL_TMPL.format(model=model_name, action=action)
        r = await client.post(url, json=body)
        r.raise_for_status()
        logging.info("[Triton] %s %s:%s accepted",
                     action, model_name, ver_str or "<policy>")

        # ② unload 或无需等待 ready
        if action == "unload" or not wait_until_ready:
            return

        # ③ 轮询 /ready
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


def load_test(test_model: str) -> bool:
    """
    真正的压测实现留空；这里先返回 True 方便主流程。
    """
    return True

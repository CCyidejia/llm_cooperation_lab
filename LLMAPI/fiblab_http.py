#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FibLab 备用 HTTP 网关（与 ``fiblab.py`` 中 ``LLMAgent`` 接口一致）

组内提供的访问方式（**不要**使用 ``.../register``，只用根地址）：

- **Base URL**：``http://35.220.164.252:3888``
- **API Key**：通过环境变量 ``FIBLAB_HTTP_API_KEY`` 设置（勿提交到公开仓库）

假定与主站相同：**OpenAI 兼容** ``POST {origin}/v1/chat/completions``，``GET {origin}/models``；
鉴权同时尝试 ``Authorization: Bearer`` 与 ``x-litellm-api-key``（以适配不同 LiteLLM 配置）。

使用方式（在实验里替换 fiblab）::

    from LLMAPI.fiblab_http import LLMAgent

环境变量：

- ``FIBLAB_HTTP_ORIGIN`` — 默认 ``http://35.220.164.252:3888``
- ``FIBLAB_HTTP_BASE_URL`` — 默认 ``{ORIGIN}/v1``
- ``FIBLAB_HTTP_API_KEY`` — 密钥
- ``FIBLAB_HTTP_DEFAULT_MODEL`` — 模型 id，建议先 ``python LLMAPI/fiblab_http.py`` 列模型
"""
from typing import Optional

import os
import json
import asyncio
import logging
import requests

_logger = logging.getLogger(__name__)

FIBLAB_HTTP_ORIGIN = os.getenv("FIBLAB_HTTP_ORIGIN", "http://35.220.164.252:3888").rstrip("/")
API_BASE_URL = os.getenv("FIBLAB_HTTP_BASE_URL", f"{FIBLAB_HTTP_ORIGIN}/v1").rstrip("/")
DEFAULT_MODEL = os.getenv("FIBLAB_HTTP_DEFAULT_MODEL", "qwen3-next-80b-a3b-instruct")
API_KEY = os.getenv(
    "FIBLAB_HTTP_API_KEY",
)
FIBLAB_HTTP_VERBOSE_REASONING_FALLBACK = os.getenv("FIBLAB_HTTP_VERBOSE_REASONING_FALLBACK", "").lower() in (
    "1",
    "true",
    "yes",
)


def fetch_fiblab_http_models(api_key: Optional[str] = None) -> dict:
    """GET /models，与主 fiblab 相同的 query 参数。"""
    key = api_key if api_key is not None else API_KEY
    url = f"{FIBLAB_HTTP_ORIGIN}/models"
    params = {
        "return_wildcard_routes": "false",
        "include_model_access_groups": "false",
        "only_model_access_groups": "false",
        "include_metadata": "false",
    }
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {key}",
        "x-litellm-api-key": key,
    }
    r = requests.get(url, params=params, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json()


def _request_headers(token: str) -> dict:
    return {
        "Content-Type": "application/json",
        "accept": "application/json",
        "Authorization": f"Bearer {token}",
        "x-litellm-api-key": token,
    }


class LLMAgent:
    """与 ``LLMAPI.fiblab.LLMAgent`` 行为一致，仅基址与默认 Key 不同。"""

    def __init__(self, name="FibLabHTTPAgent"):
        self.name = name
        self.token = API_KEY
        self.model = DEFAULT_MODEL
        self.url = f"{API_BASE_URL}/chat/completions"
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.router = self

    def get_llm_response(self, system_message: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ]
        self.total_input_tokens += self.count_tokens(messages)

        headers = _request_headers(self.token)
        model_lower = (self.model or "").lower()
        max_out = 4096 if ("r1" in model_lower or "reasoner" in model_lower or "reasoning" in model_lower) else 500

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "max_tokens": max_out,
            "temperature": 0.7,
            "top_p": 0.7,
        }

        try:
            response = requests.post(self.url, headers=headers, json=payload, timeout=300)
            response.raise_for_status()

            if not response.text or not response.text.strip():
                print(f"[{self.name}] API returned empty response")
                return "API Call Failed: Empty response from API"

            try:
                response_json = response.json()
            except json.JSONDecodeError as json_err:
                print(f"[{self.name}] Invalid JSON: {json_err}")
                return f"API Call Failed: Invalid JSON: {json_err}"

            if "choices" not in response_json or not response_json["choices"]:
                print(f"[{self.name}] No choices in API response")
                return "API Call Failed: No choices in API response"

            msg = response_json["choices"][0].get("message") or {}
            raw_content = msg.get("content")
            reply = (raw_content or "").strip() if isinstance(raw_content, str) else ""
            if not reply:
                for key in ("reasoning_content", "reasoning", "thinking"):
                    alt = msg.get(key)
                    if isinstance(alt, str) and alt.strip():
                        reply = alt.strip()
                        msg_fb = f"[{self.name}] Using message['{key}'] (content was empty)."
                        _logger.debug(msg_fb)
                        if FIBLAB_HTTP_VERBOSE_REASONING_FALLBACK:
                            print(msg_fb)
                        break

            if not reply:
                print(f"[{self.name}] Empty content; message keys: {list(msg.keys())}")
                return "API Call Failed: Empty content in API response"

            self.total_output_tokens += self.count_tokens([{"role": "assistant", "content": reply}])
            return reply

        except requests.exceptions.HTTPError as http_err:
            print(f"[{self.name}] HTTP Error: {http_err}")
            print(f"Response: {response.text if 'response' in locals() else 'N/A'}")
            return f"API Call Failed: HTTP Error: {http_err}, Response: {response.text if 'response' in locals() else ''}"
        except Exception as e:
            print(f"[{self.name}] API Call Failed: {e}")
            return f"API Call Failed: {str(e)}"

    async def acompletion(self, model=None, messages=None, stream=False):
        system_message = ""
        user_message = ""
        if messages:
            for msg in messages:
                if msg.get("role") == "system":
                    system_message = msg.get("content", "")
                elif msg.get("role") == "user":
                    user_message = msg.get("content", "")

        max_retries = 5
        retry_interval = 1

        for attempt in range(max_retries):
            try:
                loop = asyncio.get_event_loop()
                reply = await loop.run_in_executor(
                    None,
                    lambda: self.get_llm_response(system_message, user_message),
                )
                if reply.startswith("API Call Failed"):
                    if attempt < max_retries - 1:
                        print(f"[{self.name}] retry {attempt + 1}/{max_retries}...")
                        await asyncio.sleep(retry_interval)
                        continue
                    return type(
                        "obj",
                        (object,),
                        {
                            "choices": [
                                type(
                                    "obj",
                                    (object,),
                                    {"message": type("obj", (object,), {"content": reply})},
                                )
                            ]
                        },
                    )
                return type(
                    "obj",
                    (object,),
                    {
                        "choices": [
                            type(
                                "obj",
                                (object,),
                                {"message": type("obj", (object,), {"content": reply})},
                            )
                        ]
                    },
                )
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_interval)
                    continue
                err = f"API Call Failed: {str(e)}"
                return type(
                    "obj",
                    (object,),
                    {"choices": [type("obj", (object,), {"message": type("obj", (object,), {"content": err})})]},
                )

    def count_tokens(self, messages):
        total_chars = 0
        for m in messages:
            total_chars += len(m.get("role", "")) + len(m.get("content", ""))
        return total_chars // 2

    def report_token_usage(self):
        print(f"[{self.name}] Total Input Tokens: {self.total_input_tokens}")
        print(f"[{self.name}] Total Output Tokens: {self.total_output_tokens}")


if __name__ == "__main__":
    try:
        data = fetch_fiblab_http_models()
        for m in data.get("data") or []:
            print(m.get("id", "?"))
    except Exception as e:
        print(f"拉取模型列表失败: {e}")

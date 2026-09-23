#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ZGC LLM 网关（OpenAI 兼容）
==========================

- **Chat Completions 基址**：``https://zgc.apihy.com/v1``（``POST .../chat/completions``）
- **API Key**：通过环境变量 ``ZGC_LLM_API_KEY`` 设置。含真实密钥的修改勿提交到公开仓库。

在实验代码中与 ``wuwen.LLMAgent`` / ``fiblab.LLMAgent`` 互换::

    from LLMAPI.zgc import LLMAgent
    agent = LLMAgent()  # 或使用 LLMAgent(api_key="your-api-key", model="模型id")

**环境变量（可选）**

- ``ZGC_LLM_BASE_URL`` — 默认 ``https://zgc.apihy.com/v1``
- ``ZGC_LLM_API_KEY`` — 若设置则优先于文件内 ``_ZGC_API_KEY_IN_FILE``
- ``ZGC_DEFAULT_MODEL`` — 模型 id；也可在本文件 ``_ZGC_MODEL_IN_FILE`` 填写默认值（环境变量优先）。可先 ``fetch_zgc_models()`` 或 ``python -m LLMAPI.zgc`` 核对 id
**推理模型说明**：本适配器只返回 ``message.content``。即使网关还返回
``reasoning_content``，该字段也不会进入下游实验决策解析。
"""
from typing import Optional

import os
import json
import asyncio
import logging
import requests

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 默认 API Key：把密钥填在下方引号内；留空则仅依赖环境变量 ZGC_LLM_API_KEY。
# 环境变量 ZGC_LLM_API_KEY 若已设置，会覆盖此处（便于本地写死、服务器用 env）。
# ---------------------------------------------------------------------------
_ZGC_API_KEY_IN_FILE = ""

# ---------------------------------------------------------------------------
# 默认模型 id：与网关 GET /v1/models 返回的 data[].id 一致；环境变量 ZGC_DEFAULT_MODEL 可覆盖。
# ---------------------------------------------------------------------------
_ZGC_MODEL_IN_FILE = "glm-5.1"

API_BASE_URL = os.getenv("ZGC_LLM_BASE_URL", "https://zgc.apihy.com/v1").rstrip("/")
DEFAULT_MODEL = os.getenv("ZGC_DEFAULT_MODEL", _ZGC_MODEL_IN_FILE).strip()
API_KEY = os.getenv("ZGC_LLM_API_KEY", _ZGC_API_KEY_IN_FILE).strip()

# ZGC_MAX_TOKENS defaults to 16384. Set it to 0 to omit the client-side
# max_tokens field; the gateway and model will still enforce their own limits.
# ZGC_REQUEST_TIMEOUT defaults to 600 seconds. Set it to 0 for no client timeout.


def fetch_zgc_models(api_key: Optional[str] = None) -> dict:
    """
    GET /v1/models，返回 JSON（含 data[].id）。OpenAI 兼容风格。
    """
    key = api_key if api_key is not None else API_KEY
    if not key:
        raise ValueError("未设置 API Key：请在 zgc.py 中填写 _ZGC_API_KEY_IN_FILE，或设置环境变量 ZGC_LLM_API_KEY")
    url = f"{API_BASE_URL}/models"
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {key}",
    }
    r = requests.get(url, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json()


def _resolve_max_tokens(value: Optional[int] = None) -> int:
    """Return the client output budget; zero means omit max_tokens from the request."""
    raw = str(value if value is not None else os.getenv("ZGC_MAX_TOKENS", "16384")).strip()
    try:
        resolved = int(raw)
    except ValueError as exc:
        raise ValueError("ZGC_MAX_TOKENS must be a non-negative integer") from exc
    if resolved < 0:
        raise ValueError("ZGC_MAX_TOKENS must be a non-negative integer")
    return resolved


def _resolve_request_timeout(value: Optional[float] = None) -> Optional[float]:
    """Return request timeout seconds; zero explicitly opts into no client timeout."""
    raw = str(value if value is not None else os.getenv("ZGC_REQUEST_TIMEOUT", "600")).strip()
    try:
        resolved = float(raw)
    except ValueError as exc:
        raise ValueError("ZGC_REQUEST_TIMEOUT must be a non-negative number") from exc
    if resolved < 0:
        raise ValueError("ZGC_REQUEST_TIMEOUT must be a non-negative number")
    return None if resolved == 0 else resolved


class LLMAgent:
    """
    通用 LLM 封装：ZGC 网关，与 wuwen.LLMAgent / fiblab.LLMAgent 行为对齐。
    """

    def __init__(
        self,
        name="ZGCAgent",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        request_timeout: Optional[float] = None,
    ):
        self.name = name
        self.token = (api_key if api_key is not None else API_KEY).strip()
        self.model = (model if model is not None else DEFAULT_MODEL).strip() or DEFAULT_MODEL
        self.url = f"{API_BASE_URL}/chat/completions"
        self.max_tokens = _resolve_max_tokens(max_tokens)
        self.request_timeout = _resolve_request_timeout(request_timeout)

        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.router = self

    def get_llm_response(self, system_message: str, user_prompt: str) -> str:
        if not self.token:
            return "API Call Failed: 未填写 Key（_ZGC_API_KEY_IN_FILE / 环境变量 ZGC_LLM_API_KEY / LLMAgent(api_key=...)）"

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ]

        input_tokens = self.count_tokens(messages)
        self.total_input_tokens += input_tokens

        headers = {
            "Content-Type": "application/json",
            "accept": "application/json",
            "Authorization": f"Bearer {self.token}",
        }

        # 部分模型（如部分 Claude）不允许同时传 temperature 与 top_p，网关会返回 invalid_request_error
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": 0.7,
        }
        # GLM-5.1 enables deep thinking by default. Explicitly disable it for
        # these short, structured game decisions. The ZGC OpenAI-compatible
        # gateway forwards this model-specific field to the upstream API.
        if "glm-5.1" in self.model.lower():
            payload["thinking"] = {"type": "disabled"}
        if "deepseek-v4-pro" in self.model.lower():
            payload["enable_thinking"] = False
        if self.max_tokens > 0:
            payload["max_tokens"] = self.max_tokens

        try:
            response = requests.post(
                self.url,
                headers=headers,
                json=payload,
                timeout=self.request_timeout,
            )
            response.raise_for_status()

            if not response.text or not response.text.strip():
                print(f"[{self.name}] ZGC API returned empty response")
                return "API Call Failed: Empty response from API"

            try:
                response_json = response.json()
            except json.JSONDecodeError as json_err:
                print(f"[{self.name}] Invalid JSON: {json_err}")
                print(f"Response content: {response.text[:500]}")
                return f"API Call Failed: Invalid JSON: {json_err}"

            if "choices" not in response_json or not response_json["choices"]:
                print(f"[{self.name}] No choices in API response")
                return "API Call Failed: No choices in API response"

            choice = response_json["choices"][0]
            msg = choice.get("message") or {}
            finish_reason = choice.get("finish_reason", "unknown")

            # A length-limited response may contain only partial reasoning and
            # no usable final decision. Report it as a failed call so the
            # experiment-level retry loop can request a fresh completion.
            if finish_reason == "length":
                return (
                    "API Call Failed: Model output was truncated "
                    "(finish_reason=length); retrying is required"
                )

            raw_content = msg.get("content")
            if isinstance(raw_content, str):
                reply = raw_content.strip()
            elif isinstance(raw_content, list):
                parts = []
                for item in raw_content:
                    if isinstance(item, dict):
                        value = item.get("text") or item.get("content") or ""
                        if isinstance(value, str):
                            parts.append(value)
                    elif isinstance(item, str):
                        parts.append(item)
                reply = "".join(parts).strip()
            else:
                reply = ""

            if not reply:
                refusal = msg.get("refusal")
                if isinstance(refusal, str) and refusal.strip():
                    return f"API Call Failed: Model refusal: {refusal.strip()}"
                print(f"[{self.name}] Empty content in API response; finish_reason={finish_reason}; message keys: {list(msg.keys())}")
                return (
                    "API Call Failed: Empty message.content in API response; "
                    f"finish_reason={finish_reason}"
                )

            output_tokens = self.count_tokens([{"role": "assistant", "content": reply}])
            self.total_output_tokens += output_tokens

            return reply

        except requests.exceptions.HTTPError as http_err:
            print(f"[{self.name}] ZGC API HTTP Error: {http_err}")
            print(f"Response content: {response.text if 'response' in locals() else 'No response object'}")
            return f"API Call Failed: HTTP Error: {http_err}, Response: {response.text if 'response' in locals() else 'No response'}"
        except Exception as e:
            print(f"[{self.name}] ZGC API Call Failed: {e}")
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
                        print(
                            f"[{self.name}] API call failed, retrying in {retry_interval}s... "
                            f"({attempt + 1}/{max_retries})"
                        )
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
                                    {
                                        "message": type(
                                            "obj",
                                            (object,),
                                            {"content": reply},
                                        )
                                    },
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
                                {
                                    "message": type(
                                        "obj",
                                        (object,),
                                        {"content": reply},
                                    )
                                },
                            )
                        ]
                    },
                )
            except Exception as e:
                if attempt < max_retries - 1:
                    print(
                        f"[{self.name}] Exception: {e}, retrying in {retry_interval}s... ({attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(retry_interval)
                    continue
                error_msg = f"API Call Failed: {str(e)}"
                print(f"[{self.name}] {error_msg}")
                return type(
                    "obj",
                    (object,),
                    {
                        "choices": [
                            type(
                                "obj",
                                (object,),
                                {
                                    "message": type(
                                        "obj",
                                        (object,),
                                        {"content": error_msg},
                                    )
                                },
                            )
                        ]
                    },
                )

    def count_tokens(self, messages):
        total_chars = 0
        for m in messages:
            total_chars += len(m.get("role", "")) + len(m.get("content", ""))
        return total_chars // 2

    def report_token_usage(self):
        print(f"[{self.name}] Total Input Tokens: {self.total_input_tokens}")
        print(f"[{self.name}] Total Output Tokens: {self.total_output_tokens}")
        total = self.total_input_tokens + self.total_output_tokens
        print(f"[{self.name}] Total Tokens Used: {total}")


if __name__ == "__main__":
    try:
        data = fetch_zgc_models()
        rows = data.get("data") or []
        print(f"ZGC 可用模型（共 {len(rows)} 个），id 用于 ZGC_DEFAULT_MODEL / LLMAgent.model：\n")
        for m in rows:
            mid = m.get("id", "?")
            print(f"  - {mid}")
    except Exception as e:
        print(f"拉取模型列表失败: {e}")

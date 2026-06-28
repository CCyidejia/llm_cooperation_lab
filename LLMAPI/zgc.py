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
- ``ZGC_VERBOSE_REASONING_FALLBACK`` — 设为 ``1`` 时，回退使用 ``reasoning_content`` 时在终端 **print**（默认不打印，仅 DEBUG 日志）

**推理模型说明**：若网关返回 ``content`` 为空、仅 ``reasoning_content`` 有内容，本库会**用推理文本作为回复**（与 fiblab / 旧版 wuwen 行为一致）。
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
_ZGC_MODEL_IN_FILE = "claude-haiku-4-5-20251001-thinking"

API_BASE_URL = os.getenv("ZGC_LLM_BASE_URL", "https://zgc.apihy.com/v1").rstrip("/")
DEFAULT_MODEL = os.getenv("ZGC_DEFAULT_MODEL", _ZGC_MODEL_IN_FILE).strip()
API_KEY = os.getenv("ZGC_LLM_API_KEY", _ZGC_API_KEY_IN_FILE).strip()
ZGC_VERBOSE_REASONING_FALLBACK = os.getenv("ZGC_VERBOSE_REASONING_FALLBACK", "").lower() in ("1", "true", "yes")


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


class LLMAgent:
    """
    通用 LLM 封装：ZGC 网关，与 wuwen.LLMAgent / fiblab.LLMAgent 行为对齐。
    """

    def __init__(self, name="ZGCAgent", api_key: Optional[str] = None, model: Optional[str] = None):
        self.name = name
        self.token = (api_key if api_key is not None else API_KEY).strip()
        self.model = (model if model is not None else DEFAULT_MODEL).strip() or DEFAULT_MODEL
        self.url = f"{API_BASE_URL}/chat/completions"

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

        model_lower = (self.model or "").lower()
        max_out = 4096 if ("r1" in model_lower or "reasoner" in model_lower or "reasoning" in model_lower) else 500

        # 部分模型（如部分 Claude）不允许同时传 temperature 与 top_p，网关会返回 invalid_request_error
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "max_tokens": max_out,
            "temperature": 0.7,
        }

        try:
            response = requests.post(self.url, headers=headers, json=payload, timeout=300)
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

            msg = response_json["choices"][0].get("message") or {}
            raw_content = msg.get("content")
            reply = (raw_content or "").strip() if isinstance(raw_content, str) else ""
            if not reply:
                for key in ("reasoning_content", "reasoning", "thinking"):
                    alt = msg.get(key)
                    if isinstance(alt, str) and alt.strip():
                        reply = alt.strip()
                        msg_fb = f"[{self.name}] Using message['{key}'] (content was empty); downstream may parse long reasoning."
                        _logger.debug(msg_fb)
                        if ZGC_VERBOSE_REASONING_FALLBACK:
                            print(msg_fb)
                        break

            if not reply:
                print(f"[{self.name}] Empty content in API response; message keys: {list(msg.keys())}")
                return "API Call Failed: Empty content in API response"

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

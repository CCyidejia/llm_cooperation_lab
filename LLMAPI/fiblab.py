#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FibLab LLM 网关（LiteLLM / OpenAI 兼容）
========================================

平台已恢复使用，与实验室约定一致：

- **服务根 URL**：``https://llmapi.fiblab.net``
- **API Key**：通过环境变量 ``FIBLAB_LLM_API_KEY`` 设置，勿提交到公开仓库。
- **鉴权**：请求头 ``x-litellm-api-key: <key>``

**查看可用模型**（与官方 curl 一致）::

    curl -X GET \\
      'https://llmapi.fiblab.net/models?return_wildcard_routes=false&include_model_access_groups=false&only_model_access_groups=false&include_metadata=false' \\
      -H 'accept: application/json' \\
      -H 'x-litellm-api-key: $FIBLAB_LLM_API_KEY'

**对话补全**：OpenAI 兼容路径 ``POST {origin}/v1/chat/completions``。

在实验代码中与 ``wuwen.LLMAgent`` 互换::

    from LLMAPI.fiblab import LLMAgent

**环境变量（可选）**

- ``FIBLAB_ORIGIN`` — 默认 ``https://llmapi.fiblab.net``
- ``FIBLAB_LLM_BASE_URL`` — 默认 ``{FIBLAB_ORIGIN}/v1``（Chat Completions 基址）
- ``FIBLAB_LLM_API_KEY`` — API Key
- ``FIBLAB_DEFAULT_MODEL`` — ``data[].id`` 中的模型名，请先 ``fetch_fiblab_models()`` 或 curl 核对
- ``FIBLAB_VERBOSE_REASONING_FALLBACK`` — 设为 ``1`` 时，回退使用 ``reasoning_content`` 时在终端 **print**（默认不打印，仅 DEBUG 日志）

**推理模型说明**：若网关返回 ``content`` 为空、仅 ``reasoning_content`` 有内容，本库会**用推理文本作为回复**（与旧版 wuwen 行为一致）。
此类提示默认写入 **DEBUG 日志**，不刷屏；需要排查时可将日志级别设为 DEBUG，或换用非「思考链」模型（如部分 ``glm-*`` 聊天回复）。
"""
from typing import Optional

import os
import json
import asyncio
import logging
import requests

_logger = logging.getLogger(__name__)

# 文档中的网关根地址（/models 挂在根域下，见上方 curl）
FIBLAB_ORIGIN = os.getenv("FIBLAB_ORIGIN", "https://llmapi.fiblab.net").rstrip("/")
# OpenAI 兼容 Chat Completions：POST .../v1/chat/completions
API_BASE_URL = os.getenv("FIBLAB_LLM_BASE_URL", f"{FIBLAB_ORIGIN}/v1").rstrip("/")
DEFAULT_MODEL = os.getenv("FIBLAB_DEFAULT_MODEL", "qwen3-235b-a22b-instruct")
API_KEY = os.getenv("FIBLAB_LLM_API_KEY")
# 为 true 时：content 空、回退 reasoning_content 会在终端打印一行（默认 false，避免每轮刷屏）
FIBLAB_VERBOSE_REASONING_FALLBACK = os.getenv("FIBLAB_VERBOSE_REASONING_FALLBACK", "").lower() in ("1", "true", "yes")


def fetch_fiblab_models(api_key: Optional[str] = None) -> dict:
    """
    调用与文档相同的 GET /models，返回 JSON（含 data[].id）。
    等价于文档中的 curl 命令。
    """
    key = api_key if api_key is not None else API_KEY
    url = f"{FIBLAB_ORIGIN}/models"
    params = {
        "return_wildcard_routes": "false",
        "include_model_access_groups": "false",
        "only_model_access_groups": "false",
        "include_metadata": "false",
    }
    headers = {
        "accept": "application/json",
        "x-litellm-api-key": key,
    }
    r = requests.get(url, params=params, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json()


class LLMAgent:
    """
    通用 LLM 封装：FibLab 网关，与 wuwen.LLMAgent 行为对齐。
    """

    def __init__(self, name="FibLabAgent"):
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

        input_tokens = self.count_tokens(messages)
        self.total_input_tokens += input_tokens

        headers = {
            "Content-Type": "application/json",
            "accept": "application/json",
            "x-litellm-api-key": self.token,
        }

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
                print(f"[{self.name}] FibLab API returned empty response")
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
                        if FIBLAB_VERBOSE_REASONING_FALLBACK:
                            print(msg_fb)
                        break

            if not reply:
                print(f"[{self.name}] Empty content in API response; message keys: {list(msg.keys())}")
                return "API Call Failed: Empty content in API response"

            output_tokens = self.count_tokens([{"role": "assistant", "content": reply}])
            self.total_output_tokens += output_tokens

            return reply

        except requests.exceptions.HTTPError as http_err:
            print(f"[{self.name}] FibLab API HTTP Error: {http_err}")
            print(f"Response content: {response.text if 'response' in locals() else 'No response object'}")
            return f"API Call Failed: HTTP Error: {http_err}, Response: {response.text if 'response' in locals() else 'No response'}"
        except Exception as e:
            print(f"[{self.name}] FibLab API Call Failed: {e}")
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
    # 本地查看可用模型 id：python -m LLMAPI.fiblab （需在 packages/agentsociety2 下或设置 PYTHONPATH）
    try:
        data = fetch_fiblab_models()
        rows = data.get("data") or []
        print(f"FibLab 可用模型（共 {len(rows)} 个），id 用于 FIBLAB_DEFAULT_MODEL / LLMAgent.model：\n")
        for m in rows:
            mid = m.get("id", "?")
            print(f"  - {mid}")
    except Exception as e:
        print(f"拉取模型列表失败: {e}")

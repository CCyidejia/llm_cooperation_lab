import os
import requests
import json
import asyncio


# 无问苍穹平台配置
API_BASE_URL = "https://cloud.infini-ai.com/maas/v1"
DEFAULT_MODEL = "deepseek-v4-pro"  # 无问苍穹模型
# 从环境变量获取API key，或使用默认值
API_KEY = os.getenv("WUWEN_API_KEY")


def _resolve_max_tokens(model_name: str) -> int:
    """输出上限；R1 等推理模型先写 reasoning，预算不足时 content 常为空。可用环境变量 WUWEN_MAX_TOKENS 覆盖。"""
    v = os.getenv("WUWEN_MAX_TOKENS")
    if v is not None and str(v).strip():
        try:
            return max(256, int(v))
        except ValueError:
            pass
    ml = (model_name or "").lower()
    if "r1" in ml or "deepseek-r1" in ml:
        return 16384
    return 4096


class LLMAgent:
    """
    一个通用的 LLM Agent 类，用于封装无问苍穹 API 的调用。
    它不包含任何特定实验的逻辑（如构建提示、解析具体动作、管理历史）。
    """
    def __init__(self, name="WuWenAgent"):
        """
        初始化 LLMAgent。
        Args:
            name (str): 智能体的名称，用于标识日志等。
        """
        self.name = name
        self.token = API_KEY
        self.model = DEFAULT_MODEL
        self.url = f"{API_BASE_URL}/chat/completions"  # 无问苍穹的chat completions API endpoint

        # Token counters
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
        # 添加 router 属性以兼容 AgentLLM 接口
        self.router = self

    def get_llm_response(self, system_message: str, user_prompt: str) -> str:
        """
        向 LLM 发送请求并获取原始响应。
        这个方法是通用的，不处理任何实验特定逻辑。
        Args:
            system_message (str): 传递给 LLM 的系统角色消息。
            user_prompt (str): 传递给 LLM 的用户提示消息（包含所有实验内容和历史）。
        Returns:
            str: LLM 的原始文本回复。如果API调用失败，返回明确的错误信息。
        """
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ]

        # 估计输入tokens
        input_tokens = self.count_tokens(messages)
        self.total_input_tokens += input_tokens

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

        max_out = _resolve_max_tokens(self.model)

        payload = {
            "model": self.model,
            "messages": messages,  # 使用构造的messages列表
            "stream": False,
            "max_tokens": max_out,
            "temperature": 0.7,
            "top_p": 0.7
        }

        try:
            # 使用requests库调用无问苍穹API，设置超时
            response = requests.post(self.url, headers=headers, json=payload, timeout=3000)
            response.raise_for_status()  # 检查HTTP错误

            # 检查响应是否包含有效的JSON数据
            if not response.text or not response.text.strip():
                error_msg = "Empty response from API"
                print(f"[{self.name}] 无问苍穹 API returned empty response")
                return f"API Call Failed: {error_msg}"

            # 安全地解析JSON响应
            try:
                response_json = response.json()
                if "choices" not in response_json or not response_json["choices"]:
                    error_msg = "No choices in API response"
                    print(f"[{self.name}] {error_msg}")
                    return f"API Call Failed: {error_msg}"
                
                msg = response_json["choices"][0].get("message") or {}
                # 普通模型：content 为最终回复。推理模型（如 deepseek-r1）常把链式思考放在 reasoning_content，
                # 若 max_tokens 偏小，可能出现 content 为空而 reasoning_content 很长。
                raw_content = msg.get("content")
                reply = (raw_content or "").strip() if isinstance(raw_content, str) else ""
                if not reply:
                    for key in ("reasoning_content", "reasoning", "thinking"):
                        alt = msg.get(key)
                        if isinstance(alt, str) and alt.strip():
                            reply = alt.strip()
                            # 推理模型常见：content 为空、正文在 reasoning_content；默认不刷屏，排查时设 WUWEN_VERBOSE_REASONING_FALLBACK=1
                            if os.getenv("WUWEN_VERBOSE_REASONING_FALLBACK", "").strip().lower() in (
                                "1",
                                "true",
                                "yes",
                            ):
                                print(
                                    f"[{self.name}] Using message['{key}'] (content was empty); "
                                    "model may be a reasoning model."
                                )
                            break

                # 确保回复不为空
                if not reply:
                    error_msg = "Empty content in API response"
                    print(f"[{self.name}] {error_msg}")
                    print(f"[{self.name}] message keys: {list(msg.keys())}")
                    return f"API Call Failed: {error_msg}"

                # 估计输出tokens
                output_tokens = self.count_tokens([{"role": "assistant", "content": reply}])
                self.total_output_tokens += output_tokens

                return reply
            except json.JSONDecodeError as json_err:
                error_msg = f"Invalid JSON response: {json_err}"
                print(f"[{self.name}] {error_msg}")
                print(f"Response content: {response.text}")
                return f"API Call Failed: {error_msg}"

        except requests.exceptions.HTTPError as http_err:
            # 处理HTTP特定错误
            print(f"[{self.name}] 无问苍穹 API HTTP Error: {http_err}")
            print(f"Response content: {response.text if 'response' in locals() else 'No response object'}")
            return f"API Call Failed: HTTP Error: {http_err}, Response: {response.text if 'response' in locals() else 'No response'}"
        except Exception as e:
            # 处理其他潜在错误
            print(f"[{self.name}] 无问苍穹 API Call Failed: {e}")
            return f"API Call Failed: {str(e)}"
            
    async def acompletion(self, model=None, messages=None, stream=False):
        """
        异步版本的acompletion方法，兼容AgentLLM接口，添加重试机制
        
        Args:
            model: 模型名称（可选）
            messages: 消息列表，包含role和content
            stream: 是否流式返回（当前不支持）
            
        Returns:
            符合AgentLLM接口要求的响应对象
        """
        # 提取system和user消息
        system_message = ""
        user_message = ""
        
        if messages:
            for msg in messages:
                if msg.get("role") == "system":
                    system_message = msg.get("content", "")
                elif msg.get("role") == "user":
                    user_message = msg.get("content", "")
        
        # 重试机制：最多重试3次，每次间隔1秒
        max_retries = 5
        retry_interval = 1
        
        for attempt in range(max_retries):
            try:
                # 使用线程池执行同步的get_llm_response方法
                loop = asyncio.get_event_loop()
                reply = await loop.run_in_executor(
                    None,
                    lambda: self.get_llm_response(system_message, user_message)
                )
                
                # 检查是否API调用失败
                if reply.startswith("API Call Failed"):
                    if attempt < max_retries - 1:
                        print(f"[{self.name}] API call failed, retrying in {retry_interval} seconds... (Attempt {attempt+1}/{max_retries})")
                        await asyncio.sleep(retry_interval)
                        continue
                    else:
                        # 最后一次尝试仍失败，返回错误信息
                        return type('obj', (object,), {
                            'choices': [
                                type('obj', (object,), {
                                    'message': type('obj', (object,), {
                                        'content': reply
                                    })
                                })
                            ]
                        })
                
                # 构建符合接口要求的响应对象
                return type('obj', (object,), {
                    'choices': [
                        type('obj', (object,), {
                            'message': type('obj', (object,), {
                                'content': reply
                            })
                        })
                    ]
                })
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[{self.name}] Exception during API call: {e}, retrying in {retry_interval} seconds... (Attempt {attempt+1}/{max_retries})")
                    await asyncio.sleep(retry_interval)
                    continue
                else:
                    # 最后一次尝试仍失败，返回错误信息
                    error_msg = f"API Call Failed: {str(e)}"
                    print(f"[{self.name}] {error_msg}")
                    return type('obj', (object,), {
                        'choices': [
                            type('obj', (object,), {
                                'message': type('obj', (object,), {
                                    'content': error_msg
                                })
                            })
                        ]
                    })

    def count_tokens(self, messages):
        """
        计算消息列表的 token 数量。
        使用简单的字符计数作为粗略估计。
        """
        total_chars = 0
        for m in messages:
            total_chars += len(m.get("role", "")) + len(m.get("content", ""))
        # 粗略估计：假设1 token ~ 2个字符
        return total_chars // 2

    def report_token_usage(self):
        """
        打印智能体本次运行的总 token 使用量。
        """
        print(f"[{self.name}] Total Input Tokens: {self.total_input_tokens}")
        print(f"[{self.name}] Total Output Tokens: {self.total_output_tokens}")
        total = self.total_input_tokens + self.total_output_tokens
        print(f"[{self.name}] Total Tokens Used: {total}")

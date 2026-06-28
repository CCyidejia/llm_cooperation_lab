import os
import requests
import json
import asyncio
# import tiktoken


# SiliconFlow platform Qwen3-8B configuration
API_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct" # Specify the Qwen3-8B model
# Ensure the API key is retrieved from an environment variable, with a fallback.
API_KEY = os.getenv("SILICON_API_KEY")

class LLMAgent:
    """
    一个通用的 LLM Agent 类，用于封装硅基流动 API 的调用。
    它不包含任何特定实验的逻辑（如构建提示、解析具体动作、管理历史）。
    """
    def __init__(self, name="GenericAgent"):
        """
        初始化 LLMAgent。
        Args:
            name (str): 智能体的名称，用于标识日志等。
        """
        self.name = name
        self.token = API_KEY
        self.model = DEFAULT_MODEL
        self.url = f"{API_BASE_URL}/chat/completions" # SiliconFlow's chat completions API endpoint

        # Remove OpenAI client initialization as we are using requests
        # self.client = OpenAI(api_key=self.token, base_url=self.base_url)

        # ✅ Token counters (Note: tiktoken might not be accurate for Qwen)
        self.total_input_tokens = 0
        self.total_output_tokens = 0

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

        # ✅ Estimate input tokens (Note: tiktoken might not be accurate for Qwen)
        input_tokens = self.count_tokens(messages)
        self.total_input_tokens += input_tokens

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages, # Use the constructed messages list directly
            "stream": False,
            "max_tokens": 500, # Increase max_tokens to handle more complex responses
            "temperature": 0.7,
            "top_p": 0.7,
            # top_k and frequency_penalty might not be universally supported or named differently.
            # Commenting them out for broader compatibility unless specified by SiliconFlow's docs.
            # "top_k": 50,
            # "frequency_penalty": 0.5
        }

        try:
            # ✅ Use the requests library to call SiliconFlow API with timeout
            response = requests.post(self.url, headers=headers, json=payload, timeout=100)
            response.raise_for_status() # Check for HTTP errors (e.g., 4xx, 5xx)

            # 检查响应是否包含有效的JSON数据
            if not response.text or not response.text.strip():
                error_msg = "Empty response from API"
                print(f"[{self.name}] SiliconFlow API returned empty response")
                return f"API Call Failed: {error_msg}"

            # 安全地解析JSON响应
            try:
                response_json = response.json()
                if "choices" not in response_json or not response_json["choices"]:
                    error_msg = "No choices in API response"
                    print(f"[{self.name}] {error_msg}")
                    return f"API Call Failed: {error_msg}"
                
                reply = response_json["choices"][0]["message"]["content"].strip()
                
                # 确保回复不为空
                if not reply:
                    error_msg = "Empty content in API response"
                    print(f"[{self.name}] {error_msg}")
                    return f"API Call Failed: {error_msg}"

                # ✅ Estimate output tokens (Note: tiktoken might not be accurate for Qwen)
                output_tokens = self.count_tokens([{"role": "assistant", "content": reply}])
                self.total_output_tokens += output_tokens

                return reply
            except json.JSONDecodeError as json_err:
                error_msg = f"Invalid JSON response: {json_err}"
                print(f"[{self.name}] {error_msg}")
                print(f"Response content: {response.text}")
                return f"API Call Failed: {error_msg}"

        except requests.exceptions.HTTPError as http_err:
            # Handle HTTP-specific errors
            print(f"[{self.name}] SiliconFlow GPT API HTTP Error: {http_err}")
            print(f"Response content: {response.text if 'response' in locals() else 'No response object'}")
            return f"API Call Failed: HTTP Error: {http_err}, Response: {response.text if 'response' in locals() else 'No response'}"
        except Exception as e:
            # Handle other potential errors
            print(f"[{self.name}] SiliconFlow GPT API Call Failed: {e}")
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
        max_retries = 3
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
        注意：tiktoken 主要用于 OpenAI 模型，对 Qwen 可能不准确。
        如果 tiktoken 失败或不适用，将使用一个简单的字符计数作为粗略估计。
        """
        try:
            # Attempt to import tiktoken and use it
            import tiktoken
            # For Qwen, you might need a specific encoding if available,
            # otherwise, gpt-3.5-turbo's encoding is a common fallback for character-based estimation.
            enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except (KeyError, ImportError):
            # Fallback if tiktoken is not installed or model encoding is not found
            print(f"Warning: tiktoken not available or encoding not found. Using fallback character count for token estimation.")
            total_chars = 0
            for m in messages:
                total_chars += len(m.get("role", "")) + len(m.get("content", ""))
            # A very rough estimate: assume 1 token ~ 4 characters for English.
            # For Chinese, it's often closer to 1 token ~ 1 character.
            # This is a heuristic and will not be accurate.
            return total_chars // 2 # A very rough compromise

        total_tokens = 0
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            total_tokens += len(enc.encode(role)) + len(enc.encode(content))
        return total_tokens

    def report_token_usage(self):
        """
        打印智能体本次运行的总 token 使用量。
        """
        print(f"[{self.name}] Total Input Tokens: {self.total_input_tokens}")
        print(f"[{self.name}] Total Output Tokens: {self.total_output_tokens}")
        total = self.total_input_tokens + self.total_output_tokens
        print(f"[{self.name}] Total Tokens Used: {total}")

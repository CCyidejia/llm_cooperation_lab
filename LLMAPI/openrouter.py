import os
import nest_asyncio

# ==========================================
# 核心修复区：必须放在所有主要导入和执行逻辑之前
# ==========================================

# 1. 解决 ThreadPoolExecutor 中调用异步框架(如litellm)时的 Event Loop 报错
nest_asyncio.apply()

# 2. 强制设置全局环境变量（供底层的 litellm 框架自动读取以解决 401 报错）
# 请将这里的占位符替换为你真实的 API Key
# OPENROUTER_API_KEY should be set in your shell or .env file, not in source.

# 如果你使用的是自定义的中转/本地 API (比如 oapi.aivue.cn)，litellm 会读取这下面两个环境变量
# OPENAI_API_KEY and OPENAI_API_BASE are optional OpenAI-compatible settings.


# ==========================================
# 正常引入其他所需库
# ==========================================
import tiktoken
import requests
import json
import concurrent.futures
import asyncio
from openai import OpenAI
import httpx

# ==========================================
# Agent 1: LLMAgent (OpenRouter / General)
# ==========================================
API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-5.2"  # 按照用户要求的模型名称
# 优先从刚设置的环境变量中读取，增加代码健壮性
OR_API_KEY = os.getenv("OPENROUTER_API_KEY") 
SITE_URL = "https://localhost"  
SITE_NAME = "AgentSociety"     

class LLMAgent:
    """
    A universal LLM Agent class that wraps API calls.
    This version is compatible with chat models (e.g., Claude, GPT series) and
    completion models, including the unique 'o1' model.
    """

    COMPLETION_MODELS = ["text-davinci-003"]
    O1_MODEL = "o1"

    def __init__(self, name="GenericAgent"):
        self.name = name
        self.model = DEFAULT_MODEL
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        # 为DeepSeek R1模型创建线程池
        self.deepseek_thread_pool = None
        if "deepseek" in self.model.lower() and "r1" in self.model.lower():
            # 设置200个线程的线程池
            self.deepseek_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=200)
            print(f"[{self.name}] Created ThreadPoolExecutor with 200 workers for DeepSeek R1 model")

    def _make_api_call(self, model, messages, is_completion=False, prompt=None):
        """实际执行API调用的内部方法 - 使用官方 requests.post 方式"""
        if is_completion:
            full_prompt = prompt if prompt else ""
            for msg in messages:
                full_prompt += msg.get("content", "")
            
            payload = {
                "model": model,
                "prompt": full_prompt,
                "max_tokens": 1000,
                "temperature": 0.7,
                "top_p": 1.0
            }
        else:
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": 1000,
                "temperature": 0.7,
                "top_p": 1.0
            }

        # 确保 API_KEY 正确读取
        actual_api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or OR_API_KEY
        
        headers = {
            "Authorization": f"Bearer {actual_api_key}",
            "HTTP-Referer": SITE_URL,
            "X-OpenRouter-Title": SITE_NAME,
            "Content-Type": "application/json"
        }

        response = requests.post(
            url=API_URL,
            headers=headers,
            data=json.dumps(payload),
            timeout=600.0
        )
        
        if response.status_code != 200:
            key_preview = actual_api_key[:8] + "..." if actual_api_key else "NONE"
            raise Exception(f"Error code: {response.status_code} - {response.text} (API Key loaded: {key_preview})")
        
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            choice = result["choices"][0]
            if "message" in choice:
                return choice["message"].get("content", "")
            elif "text" in choice:
                return choice.get("text", "")
        
        return None

    def get_llm_response(self, system_message: str, user_prompt: str) -> str:
        reply = ""
        try:
            if self.model == self.O1_MODEL:
                full_prompt = f"{system_message}\n\n{user_prompt}".strip() if system_message else user_prompt.strip()
                input_tokens = self.count_tokens(full_prompt, is_chat_messages=False)
                self.total_input_tokens += input_tokens
                messages = [{"role": "user", "content": full_prompt}]
                
                if "deepseek" in self.model.lower() and "r1" in self.model.lower() and self.deepseek_thread_pool:
                    content = self.deepseek_thread_pool.submit(
                        self._make_api_call, self.model, messages
                    ).result()
                else:
                    content = self._make_api_call(self.model, messages)
                
                reply = content.strip() if content is not None else ""

            elif self.model in self.COMPLETION_MODELS:
                full_prompt = f"{system_message}\n\n{user_prompt}".strip() if system_message else user_prompt.strip()
                input_tokens = self.count_tokens(full_prompt, is_chat_messages=False)
                self.total_input_tokens += input_tokens

                if "deepseek" in self.model.lower() and "r1" in self.model.lower() and self.deepseek_thread_pool:
                    content = self.deepseek_thread_pool.submit(
                        self._make_api_call, self.model, None, True, full_prompt
                    ).result()
                else:
                    content = self._make_api_call(self.model, None, True, full_prompt)
                
                reply = content.strip() if content is not None else ""

            else:
                messages = []
                if system_message:
                    messages.append({"role": "system", "content": system_message})
                messages.append({"role": "user", "content": user_prompt})

                input_tokens = self.count_tokens(messages, is_chat_messages=True)
                self.total_input_tokens += input_tokens

                if "deepseek" in self.model.lower() and "r1" in self.model.lower() and self.deepseek_thread_pool:
                    content = self.deepseek_thread_pool.submit(
                        self._make_api_call, self.model, messages
                    ).result()
                else:
                    content = self._make_api_call(self.model, messages)
                
                reply = content.strip() if content is not None else ""

            is_chat = self.model not in self.COMPLETION_MODELS and self.model != self.O1_MODEL
            if reply: 
                output_tokens = self.count_tokens(
                    reply if not is_chat else [{"role": "assistant", "content": reply}],
                    is_chat_messages=is_chat
                )
                self.total_output_tokens += output_tokens

            return reply

        except Exception as e:
            print(f"[{self.name}] API 调用失败：{e}")
            return f"API 调用失败: {str(e)}"

    def count_tokens(self, content_to_encode, is_chat_messages: bool) -> int:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
        except (KeyError, ImportError):
            print(f"Warning: tiktoken fallback character count used for '{self.model}'.")
            total_chars = 0
            if is_chat_messages:
                for m in content_to_encode:
                    total_chars += len(m.get("role", "")) + len(m.get("content", ""))
            else:
                total_chars = len(content_to_encode)
            return total_chars // 2

        total_tokens = 0
        if is_chat_messages:
            for m in content_to_encode:
                role = m.get("role", "")
                content = m.get("content", "")
                total_tokens += len(enc.encode(role)) + len(enc.encode(content))
        else:
            total_tokens += len(enc.encode(content_to_encode))
        return total_tokens

    def report_token_usage(self):
        print(f"[{self.name}] Total Input Tokens: {self.total_input_tokens}")
        print(f"[{self.name}] Total Output Tokens: {self.total_output_tokens}")
        total = self.total_input_tokens + self.total_output_tokens
        print(f"[{self.name}] Total Tokens Used: {total}")

    def shutdown(self):
        if self.deepseek_thread_pool:
            print(f"[{self.name}] Shutting down DeepSeek R1 thread pool...")
            self.deepseek_thread_pool.shutdown(wait=True)
            self.deepseek_thread_pool = None
            print(f"[{self.name}] DeepSeek R1 thread pool shut down successfully")


# ==========================================
# Agent 2: LlamaAgent (OpenAI SDK wrapper)
# ==========================================
# 优先从我们顶部注入的环境变量读取，保持全局一致性
AIVUE_API_KEY = os.getenv("OPENAI_API_KEY")
AIVUE_BASE_URL = os.getenv("OPENAI_API_BASE", "http://oapi.aivue.cn/v1")
MODEL_NAME = "llama2-70b"

class LlamaAgent:
    """
    针对 llama2-70b 封装的 Agent 类
    包含：模型初始化、输入处理、Token 统计、API 调用及响应解析
    """

    def __init__(self, name="Llama2Agent"):
        self.name = name
        self.model = MODEL_NAME
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        if not AIVUE_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set in the environment")
        
        # 初始化 OpenAI 客户端
        self.client = OpenAI(
            api_key=AIVUE_API_KEY,
            base_url=AIVUE_BASE_URL,
            http_client=httpx.Client(follow_redirects=True, timeout=60.0)
        )

    def count_tokens(self, messages: list) -> int:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            return sum(len(m['content']) for m in messages) // 2

        num_tokens = 0
        for message in messages:
            num_tokens += 4 
            for key, value in message.items():
                num_tokens += len(enc.encode(value))
        return num_tokens + 2 

    def get_response(self, system_prompt: str, user_prompt: str):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        input_count = self.count_tokens(messages)
        self.total_input_tokens += input_count

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000,
                top_p=1,
                stream=False
            )

            if response.choices:
                content = response.choices[0].message.content
                output_count = self.count_tokens([{"role": "assistant", "content": content}])
                self.total_output_tokens += output_count
                return content
            else:
                return "Error: 模型未返回任何内容。"

        except Exception as e:
            return f"发生错误: {str(e)}"

    def report_usage(self):
        print(f"\n--- [{self.name}] Token 使用报告 ---")
        print(f"累计输入 Token: {self.total_input_tokens}")
        print(f"累计输出 Token: {self.total_output_tokens}")
        print(f"总计 Token: {self.total_input_tokens + self.total_output_tokens}")
        print("------------------------------\n")


# ==========================================
# 验证调用逻辑
# ==========================================
if __name__ == "__main__":
    # 初始化 Agent
    agent = LlamaAgent(name="MyLlama70B")

    # 输入处理
    sys_msg = "你是一个专业的 AI 助手，请用中文回答。"
    user_msg = "请简述 Llama2-70b 模型的参数规模及其优势。"

    # API 调用
    print("正在请求 Llama2-70b 模型，请稍候...")
    reply = agent.get_response(sys_msg, user_msg)

    # 结果展示
    print("\n[模型回复]:")
    print(reply)

    # 报告 Token 消耗
    agent.report_usage()

import os
import tiktoken
from openai import OpenAI
import httpx
import re
import concurrent.futures
import asyncio

# xty.app 平台配置
API_BASE_URL = "https://hk.xty.app/v1"
DEFAULT_MODEL = os.getenv("OPENAI_DEFAULT_MODEL", "qwen3-next-80b-a3b-instruct")
API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize a separate httpx client for direct API calls,
# and the OpenAI client for standard models.
http_client = httpx.Client(
    base_url=API_BASE_URL,
    follow_redirects=True,
    headers={"Authorization": f"Bearer {API_KEY}"},
    timeout=6000.0  # 增加超时时间到60秒，以适应大模型处理时间
)

client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL,
    http_client=http_client,
)


# ---
# The rest of the code is the same as the previous versions.
# ---

class LLMAgent:
    """
    A universal LLM Agent class that wraps API calls on the xty.app platform.
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
        """实际执行API调用的内部方法"""
        if is_completion:
            response = client.completions.create(
                model=model,
                prompt=prompt,
                stream=False,
                max_tokens=1000,  # 限制输出token为1000
                temperature=0.7,
                top_p=1.0
            )
            return response.choices[0].text if response.choices and response.choices[0] and hasattr(response.choices[0], 'text') else None
        else:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
                max_tokens=1000,  # 限制输出token为1000
                temperature=0.7,
                top_p=1.0
            )
            return response.choices[0].message.content if response.choices and response.choices[0] and hasattr(response.choices[0], 'message') else None

    def get_llm_response(self, system_message: str, user_prompt: str) -> str:
        reply = ""
        try:
            # Check if the model is o1, which requires a custom call
            if self.model == self.O1_MODEL:
                # Based on the provider's info, o1 has a different interface format
                # and doesn't support 'system' role or streaming.
                # We assume a direct call with a 'prompt' key is required.
                full_prompt = f"{system_message}\n\n{user_prompt}".strip() if system_message else user_prompt.strip()

                input_tokens = self.count_tokens(full_prompt, is_chat_messages=False)
                self.total_input_tokens += input_tokens

                messages = [{"role": "user", "content": full_prompt}]
                
                # 使用线程池执行API调用（如果是DeepSeek R1模型）
                if "deepseek" in self.model.lower() and "r1" in self.model.lower() and self.deepseek_thread_pool:
                    content = self.deepseek_thread_pool.submit(
                        self._make_api_call,
                        self.model,
                        messages
                    ).result()
                else:
                    content = self._make_api_call(self.model, messages)
                
                reply = content.strip() if content is not None else ""

            # Handle 'text-davinci-003' and other completion models
            elif self.model in self.COMPLETION_MODELS:
                full_prompt = f"{system_message}\n\n{user_prompt}".strip() if system_message else user_prompt.strip()

                input_tokens = self.count_tokens(full_prompt, is_chat_messages=False)
                self.total_input_tokens += input_tokens

                # 使用线程池执行API调用（如果是DeepSeek R1模型）
                if "deepseek" in self.model.lower() and "r1" in self.model.lower() and self.deepseek_thread_pool:
                    content = self.deepseek_thread_pool.submit(
                        self._make_api_call,
                        self.model,
                        None,
                        True,
                        full_prompt
                    ).result()
                else:
                    content = self._make_api_call(self.model, None, True, full_prompt)
                
                reply = content.strip() if content is not None else ""

            # Handle all chat models
            else:
                messages = []
                if system_message:
                    messages.append({"role": "system", "content": system_message})
                messages.append({"role": "user", "content": user_prompt})

                input_tokens = self.count_tokens(messages, is_chat_messages=True)
                self.total_input_tokens += input_tokens

                # 使用线程池执行API调用（如果是DeepSeek R1模型）
                if "deepseek" in self.model.lower() and "r1" in self.model.lower() and self.deepseek_thread_pool:
                    content = self.deepseek_thread_pool.submit(
                        self._make_api_call,
                        self.model,
                        messages
                    ).result()
                else:
                    content = self._make_api_call(self.model, messages)
                
                reply = content.strip() if content is not None else ""

            # Token count logic (unchanged)
            is_chat = self.model not in self.COMPLETION_MODELS and self.model != self.O1_MODEL
            if reply:  # 只有当reply不为空时才计算输出token
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
        """
        Calculates token count.
        Note: tiktoken is designed for OpenAI models and might not be accurate for others.
        """
        try:
            if self.model == "o1":
                enc = tiktoken.get_encoding("cl100k_base")
            elif self.model == "text-davinci-003":
                enc = tiktoken.encoding_for_model("text-davinci-003")
            elif "gpt" in self.model.lower():
                enc = tiktoken.get_encoding("cl100k_base")
            elif "claude" in self.model.lower():
                enc = tiktoken.get_encoding("cl100k_base")
            else:
                enc = tiktoken.get_encoding("cl100k_base")
        except (KeyError, ImportError):
            print(
                f"Warning: tiktoken not available or encoding not found for model '{self.model}'. Using fallback character count for token estimation.")
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
        """
        Prints the total input and output token usage for the agent's run.
        """
        print(f"[{self.name}] Total Input Tokens: {self.total_input_tokens}")
        print(f"[{self.name}] Total Output Tokens: {self.total_output_tokens}")
        total = self.total_input_tokens + self.total_output_tokens
        print(f"[{self.name}] Total Tokens Used: {total}")

    def shutdown(self):
        """
        关闭线程池，释放资源
        """
        if self.deepseek_thread_pool:
            print(f"[{self.name}] Shutting down DeepSeek R1 thread pool...")
            self.deepseek_thread_pool.shutdown(wait=True)
            self.deepseek_thread_pool = None
            print(f"[{self.name}] DeepSeek R1 thread pool shut down successfully")


import os
import tiktoken
from openai import OpenAI
import httpx

# ==========================================
# 1. 基础配置
# ==========================================
# 请替换为您真实的 API Key
API_KEY = os.getenv("OPENAI_API_KEY")
# 根据文档提供的 Base URL
BASE_URL = "http://oapi.aivue.cn/v1"
# 指定模型名称
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
        
        # 初始化 OpenAI 客户端
        # 使用 httpx 以便后续根据需要自定义超时或代理
        self.client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
            http_client=httpx.Client(follow_redirects=True, timeout=60.0)
        )

    def count_tokens(self, messages: list) -> int:
        """
        估算 Token 使用量 (基于 OpenAI cl100k_base 分词器)
        注意：Llama2 原生分词器略有不同，此处作为通用估算
        """
        try:
            enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # 降级方案：按字符数估算（每 2 个字符约 1 token）
            return sum(len(m['content']) for m in messages) // 2

        num_tokens = 0
        for message in messages:
            num_tokens += 4  # 每条消息的基础开销
            for key, value in message.items():
                num_tokens += len(enc.encode(value))
        return num_tokens + 2  # 助手回复的基础开销

    def get_response(self, system_prompt: str, user_prompt: str):
        """
        执行 API 调用并解析响应
        """
        # 构造消息结构
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # 统计输入 Tokens
        input_count = self.count_tokens(messages)
        self.total_input_tokens += input_count

        try:
            # 执行非流式请求
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000,
                top_p=1,
                stream=False
            )

            # 响应解析
            if response.choices:
                content = response.choices[0].message.content
                
                # 统计输出 Tokens
                output_count = self.count_tokens([{"role": "assistant", "content": content}])
                self.total_output_tokens += output_count
                
                return content
            else:
                return "Error: 模型未返回任何内容。"

        except Exception as e:
            return f"发生错误: {str(e)}"

    def report_usage(self):
        """
        打印当前的 Token 使用报告
        """
        print(f"\n--- [{self.name}] Token 使用报告 ---")
        print(f"累计输入 Token: {self.total_input_tokens}")
        print(f"累计输出 Token: {self.total_output_tokens}")
        print(f"总计 Token: {self.total_input_tokens + self.total_output_tokens}")
        print("------------------------------\n")


# ==========================================
# 2. 验证调用逻辑
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

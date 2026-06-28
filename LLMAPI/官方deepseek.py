import os
import tiktoken
from openai import OpenAI
import re # 保留re，因为解析数字通常是通用需求，如果API返回的是带数字的通用指令，可能仍需解析

# DeepSeek 平台默认配置
API_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
# API key must be provided by the environment, never hard-coded in source.
API_KEY = os.getenv("DEEPSEEK_API_KEY")


class LLMAgent:
    """
    一个通用的 LLM Agent 类，用于封装 DeepSeek API 的调用。
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
        self.base_url = API_BASE_URL
        if not self.token:
            raise ValueError("DEEPSEEK_API_KEY is not set in the environment")

        # ✅ 初始化 DeepSeek 客户端
        self.client = OpenAI(api_key=self.token, base_url=self.base_url)

        # ✅ Token 计数器
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
            str: LLM 的原始文本回复。如果API调用失败，返回错误信息。
        """
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ]

        # ✅ 估算 input token 数
        input_tokens = self.count_tokens(messages)
        self.total_input_tokens += input_tokens

        try:
            # ✅ 使用 DeepSeek SDK 调用，添加超时设置
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False,
                timeout=60.0  # 设置60秒超时
            )

            reply = response.choices[0].message.content.strip()

            # ✅ 估算 output token 数
            output_tokens = self.count_tokens([{"role": "assistant", "content": reply}])
            self.total_output_tokens += output_tokens

            return reply

        except Exception as e:
            print(f"[{self.name}] DeepSeek API 调用失败：{e}")
            return f"API 调用失败: {e}"

    def count_tokens(self, messages):
        """
        计算消息列表的 token 数量。
        """
        try:
            enc = tiktoken.encoding_for_model("gpt-3.5-turbo")  # DeepSeek 可能使用类似的编码
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")

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

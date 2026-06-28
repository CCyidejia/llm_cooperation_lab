from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMConfig:
    model_list: list[Any] = field(default_factory=list)
    agent: Any = None
    env: Any = None
    helper: Any = None


AgentLLMConfig = LLMConfig
EnvLLMConfig = LLMConfig
HelperLLMConfig = LLMConfig

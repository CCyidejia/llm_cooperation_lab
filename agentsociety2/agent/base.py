from dataclasses import dataclass
from typing import Any


@dataclass
class AgentLLM:
    router: Any
    model_name: str = ""

    async def acompletion(self, *args, **kwargs):
        if not hasattr(self.router, "acompletion"):
            raise AttributeError("AgentLLM.router does not provide acompletion")
        return await self.router.acompletion(*args, **kwargs)


class AgentBase:
    def __init__(self, id: int, profile: Any):
        self._id = id
        self._profile = profile
        self._llm = None
        self._env = None

    @property
    def id(self) -> int:
        return self._id

    async def init(self, llm: AgentLLM, env: Any = None):
        self._llm = llm
        self._env = env

    async def dump(self) -> dict:
        return {"profile": self._profile}

    async def load(self, dump_data: dict):
        self._profile = dump_data.get("profile", self._profile)

    async def ask(self, message: str, readonly: bool = True) -> str:
        raise NotImplementedError

    async def step(self, tick: int, t):
        raise NotImplementedError

    async def report_token_usage(self):
        pass

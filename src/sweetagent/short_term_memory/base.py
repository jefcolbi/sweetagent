from typing import List, Optional

from sweetagent.core import LLMChatMessage


class BaseShortTermMemory:
    def __init__(self, **kwargs):
        self.messages: List[LLMChatMessage] = []

    def add_message(self, message: LLMChatMessage):
        raise NotImplementedError

    def serialize_for_provider(self, provider: Optional[str] = None):
        raise NotImplementedError

    def clear(self):
        self.messages.clear()

    def inject_memories(self, memories: str):
        raise NotImplementedError

    def get_message(self, index: int) -> LLMChatMessage:
        return self.messages[index]

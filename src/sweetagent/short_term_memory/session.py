from typing import Optional

from sweetagent.core import LLMChatMessage
from sweetagent.short_term_memory.base import BaseShortTermMemory


class SessionMemory(BaseShortTermMemory):
    def add_message(self, message: LLMChatMessage):
        self.messages.append(message)

    def serialize_for_provider(self, provider: Optional[str] = None):
        return [message.to_dict(provider=provider) for message in self.messages]

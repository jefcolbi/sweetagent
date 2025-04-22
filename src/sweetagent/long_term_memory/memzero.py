from typing import List

try:
    from mem0 import Memory
    from mem0.configs.base import MemoryConfig
except ImportError:
    raise ImportError(
        "Error when importing mem0. Are you sure you installed mem0ai package ?"
    )
from sweetagent.long_term_memory.base import BaseLongTermMemory


class Mem0LongTermMemory(BaseLongTermMemory):
    def __init__(self, memory_config=MemoryConfig(), **kwargs):
        super().__init__(**kwargs)
        self.mem0 = Memory(config=memory_config)

    def retrieve_messages(self, query: str) -> List[str]:
        res = self.mem0.search(query, user_id=self._user_id, agent_id=self._agent_id)[
            "results"
        ]
        return [entry.content for entry in res]

    def add(self, user_message: str, llm_message: str):
        string_added = f"User: {user_message}\nAssistant: {llm_message}"
        self.mem0.add(string_added, user_id=self._user_id, agent_id=self._agent_id)

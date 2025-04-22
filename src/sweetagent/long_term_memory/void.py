from typing import List

from sweetagent.long_term_memory.base import BaseLongTermMemory


class VoidLongTermMemory(BaseLongTermMemory):
    def add(self, user_message: str, llm_message: str):
        pass

    def retrieve_messages(self, query: str) -> List[str]:
        return []

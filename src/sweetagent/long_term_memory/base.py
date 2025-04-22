from typing import List


class BaseLongTermMemory:
    def __init__(self, **kwargs):
        self._user_id: str = ""
        self._agent_id: str = ""

    def retrieve_messages(self, query: str) -> List[str]:
        raise NotImplementedError

    def add(self, user_message: str, llm_message: str):
        raise NotImplementedError

    def set_user_id(self, user_id: str):
        self._user_id = user_id

    def set_agent_id(self, agent_id: str):
        self._agent_id = agent_id

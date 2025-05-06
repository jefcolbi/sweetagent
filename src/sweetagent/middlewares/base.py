from sweetagent.core import ToolCall, LLMChatMessage


class BaseMiddleware:
    def after_user_message(
        self,
        llm_agent: "LLMAgent",  # noqa: F821
        llm_client: "LLMClient",  # noqa: F821
        chat_message: LLMChatMessage,
    ) -> LLMChatMessage:
        return chat_message

    def after_tool_output(
        self,
        llm_agent: "LLMAgent",  # noqa: F821
        llm_client: "LLMClient",  # noqa: F821
        tool_call: ToolCall,
        chat_message: LLMChatMessage,
    ) -> LLMChatMessage:
        return chat_message

    def after_agent_message(
        self,
        llm_agent: "LLMAgent",  # noqa: F821
        llm_client: "LLMClient",  # noqa: F821
        chat_message: LLMChatMessage,
    ) -> LLMChatMessage:
        return chat_message

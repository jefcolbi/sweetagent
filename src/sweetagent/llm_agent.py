from pydantic_core import ValidationError

from sweetagent.core import ToolCall, WorkMode, RetryToFix
from sweetagent.llm_client import LLMChatMessage, LLMClient
from typing import Callable, Optional, Dict, List
import inspect
import re

from sweetagent.short_term_memory.base import BaseShortTermMemory
from sweetagent.short_term_memory.session import SessionMemory
from sweetagent.long_term_memory.base import BaseLongTermMemory
from sweetagent.long_term_memory.void import VoidLongTermMemory
from sweetagent.utils import py_function_to_tool
from sweetagent.prompt import PromptEngine, BasePromptEngine
from sweetagent.io.base import BaseStaIO
from sweetagent.middlewares.base import BaseMiddleware


class LLMAgent:
    rgx_answer_format = re.compile(r"\[\[\s*(\w+)\s*]]")
    rgx_real_content = re.compile(
        r"\[\[\s*\w+\s*]](.*)\[\[\s*\w+\s*]]", flags=re.MULTILINE | re.DOTALL
    )

    def __init__(
        self,
        name: str,
        role: str,
        llm_client: LLMClient,
        stdio: BaseStaIO,
        short_term_memory: Optional[BaseShortTermMemory] = None,
        steps: Optional[List[str]] = None,
        instructions: Optional[str] = None,
        prompt_engine: Optional[BasePromptEngine] = None,
        native_tool_call_support: bool = True,
        user_full_name: str = "Anonymous",
        user_extra_infos: Optional[dict] = None,
        work_mode: WorkMode = WorkMode.TASK,
        long_term_memory: Optional[BaseLongTermMemory] = None,
        auto_save_in_long_term_memory: bool = False,
        auto_use_memories: bool = False,
        after_tool_output_middlewares: Optional[List[BaseMiddleware]] = None,
        after_user_message_middlewares: Optional[List[BaseMiddleware]] = None,
        after_agent_message_middlewares: Optional[List[BaseMiddleware]] = None,
        **kwargs,
    ):
        self.sta_stdio: BaseStaIO = stdio
        self.name: str = name
        self.role: str = role
        self.steps: List[str] = steps if steps is not None else []
        self.instructions: Optional[str] = instructions
        self.native_tool_call_support: bool = native_tool_call_support
        self.user_full_name: str = user_full_name
        self.user_extra_infos: Optional[dict] = user_extra_infos
        self.work_mode: WorkMode = work_mode
        self.auto_save_in_long_term_memory: bool = auto_save_in_long_term_memory
        self.auto_use_memories = auto_use_memories

        self.prompt_engine: PromptEngine = (
            prompt_engine if prompt_engine else PromptEngine()
        )
        self.prompt_engine.agent_role = self.role
        self.prompt_engine.agent_name = self.name
        self.prompt_engine.agent_steps = self.steps
        self.prompt_engine.native_tool_call_support = self.native_tool_call_support
        self.prompt_engine.user_extra_infos = self.user_extra_infos
        self.prompt_engine.user_full_name = self.user_full_name
        self.prompt_engine.agent_work_mode = work_mode

        self.llm_client: LLMClient = llm_client
        self.tools: Dict[str, Callable] = {}
        self.agents: Dict[str, "LLMAgent"] = {}
        self.configure_tools()

        self.after_user_message_middlewares = after_user_message_middlewares or []
        self.after_agent_message_middlewares = after_agent_message_middlewares or []
        self.after_tool_output_middlewares = after_tool_output_middlewares or []

        if short_term_memory:
            self.short_term_memory: BaseShortTermMemory = short_term_memory
            if len(self.short_term_memory.messages) < 1:
                self.short_term_memory.add_message(
                    LLMChatMessage(
                        role="system",
                        content=self.prompt_engine.get_system_message(
                            with_tools=(
                                None
                                if self.native_tool_call_support
                                else self.get_all_tools_for_llm()
                            )
                        ),
                    )
                )
        else:
            self.short_term_memory = SessionMemory()
            self.short_term_memory.add_message(
                LLMChatMessage(
                    role="system",
                    content=self.prompt_engine.get_system_message(
                        with_tools=(
                            None
                            if self.native_tool_call_support
                            else self.get_all_tools_for_llm()
                        )
                    ),
                )
            )

        self.long_term_memory: BaseLongTermMemory = (
            long_term_memory or VoidLongTermMemory()
        )
        self.long_term_memory.set_agent_id(self.name)
        self.long_term_memory.set_user_id(self.user_full_name)

    def configure_tools(self):
        pass

    def reset_short_term_memory(self):
        self.short_term_memory.clear()
        self.short_term_memory.add_message(
            LLMChatMessage(
                role="system",
                content=self.prompt_engine.get_system_message(
                    with_tools=(
                        None
                        if self.native_tool_call_support
                        else self.get_all_tools_for_llm()
                    )
                ),
            )
        )

    def register_function_as_tool(self, function: Callable):
        self.tools[function.__name__] = function

    def register_agent_as_tool(self, agent: "LLMAgent"):
        self.agents[agent.__class__.__name__] = agent

    def execute_tool(self, tool_call: ToolCall):
        if tool_call.name in self.tools:
            function = self.tools[tool_call.name]
            res = function(**tool_call.arguments)
            return LLMChatMessage(
                role="tool",
                content=res,
                name=tool_call.name,
                type="function_call_output",
                tool_call_id=tool_call.tool_call_id,
            )
        elif tool_call.name in self.agents:
            agent = self.agents[tool_call.name]
            res = agent.run(**tool_call.arguments)
            return LLMChatMessage(
                role="tool",
                content=res,
                name=tool_call.name,
                type="function_call_output",
                tool_call_id=tool_call.tool_call_id,
            )
        else:
            return LLMChatMessage(
                role="user", content=f"No tool with name `{tool_call.name}` was found"
            )

    def get_all_tools_for_llm(self):
        tools = list(self.tools.values())
        converted_tools = [py_function_to_tool(tool) for tool in tools]
        for agent_name, agent in self.agents.items():
            agent_as_tool = py_function_to_tool(agent.run)
            agent_as_tool["function"]["name"] = agent_name
            agent_as_tool["function"]["description"] = inspect.getdoc(agent)
            converted_tools.append(agent_as_tool)
        return converted_tools

    def run(
        self,
        query_or_task: str = None,
        use_memories: bool = False,
        save_in_memories: bool = False,
        **kwargs,
    ):
        if query_or_task is None:
            raise ValueError("Pass a valid value for query_or_task argument")

        self._check_arguments(**kwargs)

        if use_memories or self.auto_use_memories:
            memories = self.long_term_memory.retrieve_messages(f"User: {query_or_task}")
            self.sta_stdio.log_info(f"{memories = }")
            self.short_term_memory.inject_memories(
                self.prompt_engine.format_memories(memories)
            )

        self._pre_run(query_or_task=query_or_task, **kwargs)

        while True:
            llm_message = self.llm_client.complete(
                self.short_term_memory.serialize_for_provider(),
                self.get_all_tools_for_llm(),
                **self.get_client_completion_kwargs(),
            )
            self.sta_stdio.log_info(str(llm_message))

            if llm_message.tool_calls:
                pass
            elif llm_message.content:
                try:
                    llm_message = self.prompt_engine.extract_formatted_llm_response(
                        llm_message.content
                    )
                except RetryToFix as e:
                    self.short_term_memory.add_message(
                        LLMChatMessage(role="user", content=str(e))
                    )
                    continue
                except ValidationError as e:
                    self.short_term_memory.add_message(
                        LLMChatMessage(
                            role="user", content=f"Wrong Response format \n{e}"
                        )
                    )
                    continue

            self.short_term_memory.add_message(llm_message)

            if llm_message.tool_calls:
                for tool_call in llm_message.tool_calls:
                    tool_res = self.execute_tool(tool_call)
                    self.sta_stdio.log_info(f"tool ======> {tool_res}")
                    self.short_term_memory.add_message(
                        self.apply_after_tool_output_middlewares(tool_call, tool_res)
                    )
                    to_add = self.prompt_engine.get_message_to_add_to_tool_output(
                        tool_res.content
                    )
                    if to_add:
                        self.short_term_memory.add_message(to_add)
                continue
            elif llm_message.content:
                kind = llm_message.kind.lower()
                real_content = llm_message.content

                if not real_content:
                    raise ValueError(
                        f"Unable to get the real content ({real_content}) from {llm_message.content}"
                    )

                if kind == "final_answer":
                    self.reset_short_term_memory()
                    if save_in_memories or self.auto_save_in_long_term_memory:
                        self.long_term_memory.add(
                            query_or_task,
                            real_content,
                        )
                    return self._post_run(real_content)
                elif kind == "message":
                    user_input = self.sta_stdio.user_input_text(
                        self.apply_after_agent_message_middlewares(llm_message).content
                    )
                    user_message = self.apply_after_user_message_middlewares(
                        LLMChatMessage(role="user", content=user_input)
                    )
                    self.short_term_memory.add_message(user_message)
                    continue
                elif kind == "tool_call":
                    pass
                else:
                    raise ValueError(f"Wrong kind found ({kind})")

    def get_real_content(self, content: str) -> str:
        r = self.rgx_real_content.search(content)
        if r:
            return r.group(1)

        parts = self.rgx_answer_format.split(content)
        if len(parts) > 1:
            return parts[1]

    def _check_arguments(self, **kwargs):
        pass

    def _pre_run(self, query_or_task: str = None, **kwargs):
        """Performs some actions before running the agent.
        By default we put the query_or_task string in the short term memory (history)"""
        self.short_term_memory.add_message(
            LLMChatMessage(
                role="user",
                content=self.prompt_engine.modify_message_before_sending(query_or_task),
            )
        )

    def _post_run(self, real_content: str) -> str:
        """Process the answer of the agent before running."""
        return real_content

    def apply_after_user_message_middlewares(
        self, to_send_chat_message: LLMChatMessage
    ) -> LLMChatMessage:
        for middleware in self.after_user_message_middlewares:
            to_send_chat_message = middleware.after_user_message(
                self, self.llm_client, to_send_chat_message
            )
        return to_send_chat_message

    def apply_after_agent_message_middlewares(
        self, to_send_chat_message: LLMChatMessage
    ) -> LLMChatMessage:
        for middleware in self.after_agent_message_middlewares:
            to_send_chat_message = middleware.after_agent_message(
                self, self.llm_client, to_send_chat_message
            )
        return to_send_chat_message

    def apply_after_tool_output_middlewares(
        self, tool_call: ToolCall, to_send_chat_message: LLMChatMessage
    ) -> LLMChatMessage:
        for middleware in self.after_tool_output_middlewares:
            to_send_chat_message = middleware.after_tool_output(
                self, self.llm_client, tool_call, to_send_chat_message
            )
        return to_send_chat_message

    def get_client_completion_kwargs(self) -> dict:
        return {}

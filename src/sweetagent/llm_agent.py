from pydantic_core import ValidationError

from sweetagent.core import ToolCall, WorkMode, RetryToFix
from sweetagent.llm_client import LLMChatMessage, LLMClient
from typing import Callable, Optional, Dict, List
import inspect
import re

from sweetagent.short_term_memory.base import BaseShortTermMemory
from sweetagent.short_term_memory.session import SessionMemory
from sweetagent.utils import py_function_to_tool
from sweetagent.prompt import PromptEngine
from sweetagent.io import BaseStaIO


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
        short_term_memory: Optional[SessionMemory],
        stdio: BaseStaIO,
        steps: Optional[List[str]] = None,
        prompt_engine: Optional[PromptEngine] = None,
        native_tool_call_support: bool = True,
        user_full_name: str = "Anonymous",
        user_extra_infos: Optional[dict] = None,
        work_mode: WorkMode = WorkMode.TASK,
        **kwargs,
    ):
        self.sta_stdio: BaseStaIO = stdio
        self.name: str = name
        self.role: str = role
        self.steps: List[str] = steps if steps is not None else []
        self.native_tool_call_support: bool = native_tool_call_support
        self.user_full_name: str = user_full_name
        self.user_extra_infos: Optional[dict] = user_extra_infos
        self.work_mode: WorkMode = work_mode

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

    def run(self, query_or_task: str = None):
        if query_or_task is None:
            raise ValueError("Pass a valid value for query_or_task argument")

        self.short_term_memory.add_message(
            LLMChatMessage(
                role="user",
                content=self.prompt_engine.modify_message_before_sending(query_or_task),
            )
        )

        while True:
            llm_message = self.llm_client.complete(
                self.short_term_memory.serialize_for_provider(),
                self.get_all_tools_for_llm(),
            )
            self.sta_stdio.log_info(str(llm_message))

            if llm_message.tool_calls:
                pass
            elif llm_message.content:
                try:
                    cust_resp = self.prompt_engine.extract_formatted_llm_response(
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

                self.sta_stdio.log_debug(str(cust_resp))
                llm_message = LLMChatMessage(role="user", content=cust_resp.message)
                self.sta_stdio.log_info(str(llm_message))

                if cust_resp.tool_name:
                    llm_message.name = cust_resp.tool_name
                    llm_message.tool_calls = [
                        ToolCall.from_formatted_response_model(cust_resp)
                    ]

            self.short_term_memory.add_message(llm_message)

            if llm_message.tool_calls:
                for tool_call in llm_message.tool_calls:
                    tool_res = self.execute_tool(tool_call)
                    self.sta_stdio.log_info(f"tool ======> {tool_res}")
                    self.short_term_memory.add_message(tool_res)
                    if self.work_mode == WorkMode.TASK:
                        self.short_term_memory.add_message(
                            LLMChatMessage(
                                role="user",
                                content="For kind == final_answer there must be a `message` "
                                "section where you put the answer.",
                            )
                        )
                continue
            elif llm_message.content:
                kind = cust_resp.kind.lower()
                real_content = cust_resp.message

                if not real_content:
                    raise ValueError(
                        f"Unable to get the real content ({real_content}) from {llm_message.content}"
                    )

                if kind == "final_answer":
                    self.reset_short_term_memory()
                    return real_content
                elif kind == "message":
                    user_input = self.sta_stdio.user_input_text(real_content)
                    self.short_term_memory.add_message(
                        LLMChatMessage(role="user", content=user_input)
                    )
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

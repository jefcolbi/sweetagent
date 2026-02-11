import json
from typing import Optional, List, Literal, Dict
from pydantic import BaseModel
import re
from _io import StringIO
import yaml
from trender import TRender
from sweetagent.core import WorkMode, RetryToFix, LLMChatMessage, ToolCall


class FormatResponseModel(BaseModel):
    kind: Literal["message", "tool_call", "final_answer"]
    message: Optional[str] = None
    data: Optional[dict] = None
    tool_name: Optional[str] = None
    tool_arguments: Optional[dict] = None


class BasePromptEngine:
    def __init__(self, **kwargs):
        # These informations are set by the agent
        self.agent_name: str = ""
        self.agent_role: str = ""
        self.native_tool_call_support: bool = True
        self.native_thought = False
        self.agent_steps: List[str] = []
        self.user_full_name: str = ""
        self.user_extra_infos: Optional[dict] = None
        self.agent_work_mode: WorkMode = WorkMode.TASK

    def get_system_message(self, with_tools: Optional[List[dict]] = None):
        raise NotImplementedError

    def modify_message_before_sending(self, message: str) -> str:
        return message

    def format_memories(self, memories: List[str]):
        res = "\n".join([f"{i}. {entry}" for i, entry in enumerate(memories)])
        return f"-------------------------\nYour memories\n\n{res}"

    def extract_formatted_llm_response(self, text_response: str) -> LLMChatMessage:
        raise NotImplementedError

    def get_message_to_add_to_tool_output(
        self, tool_output: str
    ) -> Optional[LLMChatMessage]:
        return None


class SimplePromptEngine(BasePromptEngine):
    def get_system_message(self, with_tools: Optional[List[dict]] = None):
        res = f"""You are {self.agent_name}.

You role is {self.agent_role}.

-------------------------------

The name of user you are interacting with is: {self.user_full_name}

Here are extra informations about him/her:
{json.dumps(self.user_extra_infos, indent=2) if self.user_extra_infos else ""}

"""

        if self.agent_steps:
            res += "Here are the steps you must follow to perform your work.\n\n"
            for i, step in enumerate(self.agent_steps, start=1):
                res += f"{i}. {step}\n"

        return res

    def extract_formatted_llm_response(self, text_response: str) -> LLMChatMessage:
        return LLMChatMessage(
            role="assistant",
            content=text_response,
            kind="message" if self.agent_work_mode == WorkMode.CHAT else "final_answer",
        )


class JsonPromptEngine(BasePromptEngine):
    pass


class PromptEngine(BasePromptEngine):
    rgx_section = re.compile(r"\++\s*(\w+)\s*\++")
    rgx_tool_arguments_field = re.compile(r"~+\s*(\w+)\s*~+")

    response_format_template = TRender("""#if @native_thought:
#else:
+++ thought +++   (MANDATORY SECTION)
[ REASONING Let’s think step by step ]
#end
+++ kind +++   (MANDATORY SECTION)
#if @native_tool_support:
#if @task_mode:
[ KIND OF MESSAGE (message OR final_answer)]
#else
[ KIND OF MESSAGE (message)]
#end
#else
#if @task_mode:
[ KIND OF MESSAGE (message OR tool_call OR final_answer)]
#else
[ KIND OF MESSAGE (message OR tool_call)]
#end
#end
+++ message +++   (OPTIONAL SECTION)
[ Your message's content here ]
+++ data +++   (OPTIONAL SECTION)
[ DATA Yaml format]
#if @native_tool_support:
#else:
+++ tool_name +++   (OPTIONAL SECTION)
[TOOL NAME if kind of message = tool_call]
+++ tool_arguments +++  (OPTIONAL SECTION)
[ TOOL ARGUMENTS if kind of message = tool_call ]
#end
+++ end +++   (MANDATORY END OF FORMAT)""")

    response_simple_message_template = TRender("""#if @native_thought:
#else:
+++ thought +++
The user is asking for ... so i will send him a message
#end
+++ kind +++
message
+++ message +++
Your real message here
+++ end +++""")

    response_final_answer_template = TRender("""#if @native_thought:
#else:
+++ thought +++
The user is asking for ... so i will send him a message
#end
+++ kind +++
final_answer
+++ message +++
Your answer here
+++ end +++""")

    response_message_with_choices_template = TRender("""#if @native_thought:
#else:
+++ thought +++
Maybe the user wants to send an email. I am will ask him
#end
+++ kind +++
message
+++ message +++
Do you want me so send an email ?
+++ data +++
choices:
  - Yes
  - No
+++ end +++""")

    response_tool_call_template = TRender("""#if @native_thought:
#else:
+++ thought +++
The user wants to know the max credit he can have. His username is john and age is 30.
There is a tool get_user_max_credit. I will use this tool to get the max credit.
#end
+++ kind +++
tool_call
+++ tool_name +++
get_user_max_credit
+++ tool_arguments +++
~~~ username ~~~
john

~~~ age ~~~
30

+++ end +++""")

    def get_system_message(self, with_tools: Optional[List[dict]] = None):
        res = f"""You are {self.agent_name}.

You role is {self.agent_role}.

-------------------------------

The name of user you are interacting with is: {self.user_full_name}

Here are extra informations about him/her:
{json.dumps(self.user_extra_infos, indent=2) if self.user_extra_infos else ""}

-------------------------------

STRICTLY use this TEXT format when returning your response to user.
{self.get_llm_response_format()}

-------------------------------

Here are some examples of the format usage:

Simple message
{self.get_llm_response_simple_message_format()}

############################

Question with choices
{self.get_llm_response_question_with_choices_format()}

"""

        if not self.native_tool_call_support and with_tools:
            res += f"""#############################

External tool calling
{self.get_llm_response_tool_call_format()}

#############################

Final answer after tools call
{self.get_llm_response_final_answer_format()}

"""

            res += (
                f"-------------------------------------------\n\nList of tools available\n\n"
                f"{json.dumps(with_tools, indent=2)}\n\n--------------------------------------\n\n"
            )

        if self.agent_steps:
            res += "Here are the steps you must follow to perform your work.\n\n"
            for i, step in enumerate(self.agent_steps, start=1):
                res += f"{i}. {step}\n"

        return res

    def get_llm_response_format(self):
        return self.response_format_template.render(
            {
                "native_thought": self.native_thought,
                "native_tool_support": self.native_tool_call_support,
                "task_mode": True if self.agent_work_mode == WorkMode.TASK else False,
                "chat_mode": True if self.agent_work_mode == WorkMode.CHAT else False,
            }
        )

    def get_llm_response_simple_message_format(self):
        return self.response_simple_message_template.render(
            {
                "native_thought": self.native_thought,
                "native_tool_support": self.native_tool_call_support,
            }
        )

    def get_llm_response_final_answer_format(self):
        return self.response_final_answer_template.render(
            {
                "native_thought": self.native_thought,
                "native_tool_support": self.native_tool_call_support,
            }
        )

    def get_llm_response_question_with_choices_format(self):
        return self.response_message_with_choices_template.render(
            {
                "native_thought": self.native_thought,
                "native_tool_support": self.native_tool_call_support,
            }
        )

    def get_llm_response_tool_call_format(self):
        return self.response_tool_call_template.render(
            {
                "native_thought": self.native_thought,
                "native_tool_support": self.native_tool_call_support,
            }
        )

    def extract_formatted_llm_response(self, text_response: str) -> LLMChatMessage:
        lines = text_response.splitlines(keepends=True)
        current_section: str = None
        string_builder: StringIO = None
        sections = {}

        for line in lines:
            r = self.rgx_section.search(line)
            if r:
                if current_section:
                    sections[current_section] = string_builder.getvalue().strip()
                    string_builder.close()
                    string_builder = None

                section_name = r.group(1).lower()
                if section_name == "end":
                    break
                else:
                    current_section = section_name
                    string_builder = StringIO()
                    continue
            else:
                if string_builder:
                    string_builder.write(line)

        if "data" in sections and sections["data"]:
            sections["data"] = self._decode_data_section(sections["data"])

        if "tool_arguments" in sections and sections["tool_arguments"]:
            sections["tool_arguments"] = self._decode_tool_arguments_section(
                sections["tool_arguments"]
            )

        if not sections.get("data"):
            sections["data"] = None

        if not sections.get("tool_name"):
            sections["tool_name"] = None

        if not sections.get("tool_arguments"):
            sections["tool_arguments"] = None

        sections.pop("thought", None)

        if sections["tool_name"]:
            tool_call = ToolCall(
                name=sections["tool_name"],
                type="function",
                arguments=sections["tool_arguments"],
            )
        else:
            tool_call = None

        kind = (
            sections.get("kind")
            or ("message" if self.agent_work_mode == WorkMode.CHAT else "final_answer")
        ).lower()

        res = LLMChatMessage(
            role="assistant",
            content=sections.get("message"),
            data=sections["data"],
            tool_calls=[tool_call] if tool_call else None,
            kind=kind,
        )

        if res.kind == "final_answer" and not (res.content or res.data):
            raise RetryToFix(
                "For kind == final_answer there must be a `message` or `data` section where you put the answer."
            )

        return res

    def _decode_data_section(self, data_content: str):
        return yaml.safe_load(data_content)

    def _decode_tool_arguments_section(self, tool_arguments_content: str):
        lines = tool_arguments_content.splitlines(keepends=True)
        current_field: str = None
        string_builder: StringIO = None
        fields = {}

        for line in lines:
            r = self.rgx_tool_arguments_field.search(line)
            if r:
                if current_field:
                    fields[current_field] = string_builder.getvalue().strip()
                    string_builder.close()
                    string_builder = None

                current_field = r.group(1).lower()
                string_builder = StringIO()
            else:
                if string_builder:
                    string_builder.write(line)

        if string_builder:
            fields[current_field] = string_builder.getvalue().strip()
            string_builder.close()

        return fields

    def modify_message_before_sending(self, message: str) -> str:
        return f"{message}\n\n[[ respect the response format ]]"

    def get_message_to_add_to_tool_output(
        self, tool_output: str
    ) -> Optional[LLMChatMessage]:
        if self.agent_work_mode == WorkMode.TASK:
            return LLMChatMessage(
                role="user",
                content="For kind == final_answer there must be a `message` "
                "section where you put the answer.",
            )


class BaseState:
    name: str

    def __init__(
        self,
        entry: Optional[str] = None,
        do: Optional[str] = None,
        _exit: Optional[str] = None,
    ):
        self.entry = entry
        self.do = do
        self._exit = _exit

        self.transitions: List[Dict] = []

    def add_transition(
        self,
        event: Optional[str] = None,
        condition: Optional[str] = None,
        action: Optional[str] = None,
        next_state: Optional[str] = None,
    ):
        trans = {}
        if event:
            trans["event"] = event
        if condition:
            trans["condition"] = condition
        if action:
            trans["action"] = action
        if next_state:
            trans["next_state"] = next_state
        self.transitions.append(trans)

    def to_string(self):
        res = self.build_decl()
        transs = self.build_multiple_transitions()
        if transs:
            res += f"\n\n{transs}"
        return res

    def build_decl(self):
        res = f"state {self.name}"

        if self.do or self.entry or self._exit:
            res += " {"
            if self.entry:
                res += f"\n  entry / {self.entry}()"
            if self.do:
                res += f"\n  do / {self.do}()"
            if self._exit:
                res += f"\n  exit / {self._exit}"
            res += "\n}"

        return res

    def build_single_transition(self, transition: Dict):
        event: str = transition.get("event")
        condition: str = transition.get("condition")
        action: str = transition.get("action")
        next_state: str = transition.get("next_state")

        res = f"{self.name} "
        if next_state:
            res += f"--> {next_state} "
        if event:
            res += f": {event} "
        if condition:
            res += f"[{condition}] "
        if action:
            res += f"/ {action}"
        return res

    def build_multiple_transitions(self):
        res = ""
        for trans in self.transitions:
            res += f"{self.build_single_transition(trans)}\n"
        return res

    def __str__(self):
        res = f"{self.name}("
        parts = []
        if self.entry:
            parts.append(f"entry={self.entry}")
        if self.do:
            parts.append(f"do={self.do}")
        if self._exit:
            parts.append(f"exit={self._exit}")
        if self.transitions:
            parts.append(f"transitions={self.transitions}")
        res += ", ".join(parts)
        res += ")"
        return res


class FSM:
    def __init__(
        self, initial_state: BaseState, end_state: BaseState, states: List[BaseState]
    ):
        pass


class FSMPromptEngine(BasePromptEngine):
    rgx_plus = re.compile(r"\++")

    def __init__(self, **kwargs):
        self.initial_state: BaseState = kwargs.pop("initial_state")
        self.end_state: BaseState = kwargs.pop("end_state")
        self.states: List[BaseState] = kwargs.pop("states")
        super().__init__(**kwargs)

    def get_system_message(self, with_tools: Optional[List[dict]] = None):
        states_as_str = ""
        for state in self.states:
            states_as_str += f"{state.to_string()}\n\n"

        return f"""You are a conversational agent named {self.agent_name} driven STRICTLY by a state machine.

ABSOLUTE RULES:

1. You permanently maintain an internal variable `current_state`.
2. You may produce ONLY responses authorized by the current state.
3. You may change state ONLY through an explicitly valid transition.
4. If the user's response is invalid for the current state, you MUST remain in the same state and repeat the question.
5. You NEVER anticipate future information.
6. You NEVER ask for multiple pieces of information at once.
7. You NEVER explain the internal logic or the states.

ROLE:

{self.agent_role}

FINITE STATE MACHINE (UNIQUE AUTHORITY):

```pl
@startuml
[*] --> {self.initial_state.name}

{states_as_str}
{self.end_state.name} --> [*]
@enduml
```

FORMAT TO REPLY TO USER:

```
[Your message...]
+++++++
[current_state]
```
"""

    def extract_formatted_llm_response(self, text_response: str) -> LLMChatMessage:
        parts = self.rgx_plus.split(text_response)
        current_state = parts[-1].strip()
        return LLMChatMessage(
            role="assistant",
            content=parts[0].strip(),
            kind="message" if self.end_state.name != current_state else "final_answer",
        )

import json
from typing import Optional, List, Literal
from pydantic import BaseModel
import re
from _io import StringIO
import yaml
from trender import TRender
from sweetagent.core import WorkMode, RetryToFix


class FormatResponseModel(BaseModel):
    kind: Literal["message", "tool_call", "final_answer"]
    message: Optional[str] = None
    data: Optional[dict] = None
    tool_name: Optional[str] = None
    tool_arguments: Optional[dict] = None


class PromptEngine:
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
        res = f"""You are {self.agent_name}.

You role is {self.agent_role}.

-------------------------------

The name of user you are interacting with is: {self.user_full_name}

Here are extra informations about him/her:
{json.dumps(self.user_extra_infos, indent=2) if self.user_extra_infos else ""}

-------------------------------

STRICTLY use this JSON format when returning your response to user.
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

    def extract_formatted_llm_response(self, text_response: str) -> FormatResponseModel:
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
        res = FormatResponseModel(**sections)

        if res.kind == "final_answer" and not res.message:
            raise RetryToFix(
                "For kind == final_answer there must be a `message` section where you put the answer."
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

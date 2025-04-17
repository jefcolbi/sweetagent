from unittest import TestCase
from pathlib import Path
import os

cur_dir = Path(os.path.abspath(__file__)).parent
src_path = cur_dir.parent / "src"
print(f"{src_path = }")

from sweetagent.prompt import PromptEngine


class PromptEngineTestCase(TestCase):
    def test_01_system_message(self):
        engine = PromptEngine()
        engine.native_tool_call_support = False
        engine.native_thought = True
        engine.agent_name = "Support Agent"
        engine.agent_role = "Provide support to customers"
        engine.user_full_name = "Jeffersson Mattersson"
        engine.user_extra_infos = {"age": 10}
        engine.agent_steps = [
            "Ask the user where he is going",
            "Find the price for this destination",
            "Book the travel",
        ]
        print(engine.get_system_message(with_tools=[{"yes": "no"}]))

    def test_02_decode_simple_message(self):
        example = """+++ thought +++
The user is requesting a Python program to compute the Fibonacci sequence. I will provide a simple implementation using an iterative approach.
+++ kind +++
message
+++ message +++
Certainly! Below is a Python program to compute the Fibonacci sequence up to a specified number of terms:

def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]

    fib_sequence = [0, 1]
    for i in range(2, n):
        next_value = fib_sequence[-1] + fib_sequence[-2]
        fib_sequence.append(next_value)

    return fib_sequence

# Example usage
num_terms = 10
result = fibonacci(num_terms)
print(f"Fibonacci sequence with {num_terms} terms: {result}")

This program defines a function fibonacci that takes an integer n as input and returns a list containing the first n numbers of the Fibonacci sequence.
+++ data +++

+++ tool_name +++

+++ tool_arguments +++

+++ end +++"""

        engine = PromptEngine()
        engine.extract_formatted_llm_response(example)

    def test_03_decode_message_with_data(self):
        example = """+++ thought +++
Maybe the user wants to send an email. I am will ask him
+++ kind +++
message
+++ message +++
Do you want me so send an email ?
+++ data +++
choices:
  - Yes
  - No
+++ end +++"""
        engine = PromptEngine()
        engine.extract_formatted_llm_response(example)

    def test_04_decode_tool_call(self):
        example = """+++ thought +++
The user wants to know the current weather in Douala. I will use the get_weather tool to retrieve this information.
+++ kind +++
tool_call
+++ tool_name +++
get_weather
+++ tool_arguments +++
~~~~ city ~~~~
Douala

+++ end +++"""

        engine = PromptEngine()
        engine.extract_formatted_llm_response(example)

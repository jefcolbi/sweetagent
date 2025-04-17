import inspect
from typing import Callable, List
from pydantic import BaseModel, create_model


def py_function_to_tool(function: Callable) -> dict:
    signature = inspect.signature(function)
    params = signature.parameters

    fields = {}

    for name, param in params.items():
        fields[name] = (param.annotation, param.default)

    description = inspect.getdoc(function)

    ArgTempModel = create_model("ArgTempModel", **fields)
    schema = ArgTempModel().schema()
    schema.pop("title", None)

    res = {
        "type": "function",
        "function": {
            "name": function.__name__,
            "description": description,
            "parameters": schema,
        },
    }
    return res


if __name__ == "__main__":

    def my_function(arg1: int, arg2: str, *args: int, **kwargs: float) -> bool:
        """this function perform a matrix calcultation"""
        pass

    class A:
        def method(
            self,
            arr: list = None,
        ):
            pass

    a = A()

    class A_Method_Input(BaseModel):
        arr: List[str] = None

    print(A_Method_Input().schema())

    py_function_to_tool(my_function)

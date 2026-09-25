import llm_sdk.llm_sdk as llm_sdk
from .models import FunctionDefinition
from typing import Any


class Decoder:
    """To decode the prompt into an valid JSON output format."""
    def __init__(
        self, llm: llm_sdk.Small_LLM_Model,
        functions_list: list[FunctionDefinition]
        ) -> None:
        """Initialize the decoder.

        Args:
            llm: A instance of the LLM to compute with the data
            function_defintion: A list of fix functions the class work with.
        """
        self.llm = llm
        self.functions_list = functions_list

    def prompt_to_json(self, prompt: str) -> dict[str, Any]:
        pass

    def _build_system_prompt(self) -> str:
        result: str = (
            "You are a function calling assistant. Available functions:")
        for func in self.functions_list:
            params = ", ".join(
                [f"{key}: {val.type}" for key, val in func.parameters.items()])
            result += f"\n- {func.name} ({params})"
        return result

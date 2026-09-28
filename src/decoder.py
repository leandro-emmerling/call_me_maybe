import llm_sdk.llm_sdk as llm_sdk
from .models import FunctionDefinition
from typing import Any
import json
from enum import Enum


class State(Enum):
    START = 1
    FUNCTION_NAME = 2
    PARAM_KEY = 3
    PARAM_VALUE = 4


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
        with open(self.llm.get_path_to_vocab_file()) as f:
            self.vocab_dict: dict[str, int] = json.load(f)

    def prompt_to_json(self, prompt: str) -> dict[str, Any]:
        full_sys_prompt: str = (
            self._build_system_prompt() + "\nUser request:" + prompt)
        tensor = self.llm.encode(full_sys_prompt)
        input_ids: list[int] = tensor[0].tolist()
        logits: list[float] = self.llm.get_logits_from_input_ids(input_ids)
        output_json: str = ""
        current_state: State = State.START
        bracket_count: int = 1
        while (bracket_count != 0):
            if current_state == State.START:
                output_json += '{"name": "'
                current_state = State.FUNCTION_NAME
            elif current_state == State.FUNCTION_NAME:
                ...
            elif current_state == State.PARAM_KEY:
                ...
            elif current_state == State.PARAM_VALUE:
                ...

    def _build_system_prompt(self) -> str:
        sys_prompt: str = (
            "You are a function calling assistant. Available functions:")
        for func in self.functions_list:
            params = ", ".join(
                [f"{key}: {val.type}" for key, val in func.parameters.items()])
            sys_prompt += f"\n- {func.name} ({params})"
        return sys_prompt

    def get_valid_token() -> None:
        current_state = State.START
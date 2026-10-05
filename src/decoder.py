import llm_sdk.llm_sdk as llm_sdk
from .models import FunctionDefinition
from typing import Any
import json
from enum import Enum
import numpy

class State(Enum):
    START = 1
    FUNCTION_NAME = 2
    PARAM_KEY = 3
    PARAM_VALUE = 4
    PARAM_SEPERATOR = 5


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
        self.id_to_text = {v: k for k, v in self.vocab_dict.items()}

    def prompt_to_json(self, prompt: str) -> dict[str, Any]:
        full_sys_prompt: str = (
            self._build_system_prompt() + "\nUser request:" + prompt)
        tensor = self.llm.encode(full_sys_prompt)
        input_ids: list[int] = tensor[0].tolist()
        logits: list[float] = self.llm.get_logits_from_input_ids(input_ids)
        current_state: State = State.START
        bracket_count: int = 1
        current_func_name: str = ""
        current_param_index: int = 0
        current_value: str = ""
        selected_func: FunctionDefinition | None = None
        current_param_key: str = ""
        collected_params: dict[str, Any] = {}
        while (bracket_count != 0):

            if current_state == State.START:
                fixed_start: str = '{"name": "'
                start_ids = self.llm.encode(fixed_start)[0].tolist()
                input_ids += start_ids
                output_json: str = fixed_start
                current_state = State.FUNCTION_NAME

            elif current_state == State.FUNCTION_NAME:
                logits = self.llm.get_logits_from_input_ids(input_ids)
                modified_logits: list[float] = self._modify_logits(
                    logits, self._valid_token_ids(current_func_name))
                highest_index: int = numpy.argmax(modified_logits)
                current_func_name += self.id_to_text[highest_index]
                input_ids.append(highest_index)
                if current_func_name in (
                    [func.name for func in self.functions_list]
                    ):
                    fixed_params: str = '", "parameters": {'
                    param_ids = self.llm.encode(fixed_params)[0].tolist()
                    input_ids += param_ids
                    output_json += current_func_name + fixed_params
                    bracket_count += 1
                    selected_func = next(
                        f for f in self.functions_list if (
                            f.name == current_func_name))
                    current_state = State.PARAM_KEY

            elif current_state == State.PARAM_KEY:
                param_key, param_type = list(
                    selected_func.parameters.items())[current_param_index]
                current_param_key = param_key
                param_keys: str = f'"{param_key}": '
                param_keys_ids = self.llm.encode(param_keys)[0].tolist()
                input_ids += param_keys_ids
                output_json += param_keys
                current_state = State.PARAM_VALUE

            elif current_state == State.PARAM_VALUE:
                logits = self.llm.get_logits_from_input_ids(input_ids)
                valid_num_tokens: list[int] = self._valid_num_token_ids()
                if not valid_num_tokens:
                    current_state = State.PARAM_SEPERATOR
                else:
                    modified_logits: list[float] = self._modify_logits(
                        logits, valid_num_tokens)
                    highest_index: int = numpy.argmax(modified_logits)
                    current_value += self.id_to_text[highest_index]
                    input_ids.append(highest_index)

            elif current_state == State.PARAM_SEPERATOR:
                output_json += current_value
                output_sign: str = ""
                collected_params[current_param_key] = current_value
                if current_param_index < len(selected_func.parameters) - 1:
                    output_sign += ", "
                    current_param_index += 1
                    current_value = ""
                    current_state = State.PARAM_KEY
                else:
                    output_sign = "}}"
                    bracket_count -= 2
                output_json += output_sign
                end_ids = self.llm.encode(output_sign)[0].tolist()
                input_ids += end_ids
        return {
            "prompt": prompt,
            "name": current_func_name,
            "parameters": collected_params
        }


    def _build_system_prompt(self) -> str:
        sys_prompt: str = (
            "You are a function calling assistant. Available functions:")
        for func in self.functions_list:
            params = ", ".join(
                [f"{key}: {val.type}" for key, val in func.parameters.items()])
            sys_prompt += f"\n- {func.name} ({params})"
        return sys_prompt

    def get_valid_token(self) -> None:
        current_state = State.START

    def _check_next_token(self, current_string: str, next_token: str) -> bool:
        for func in self.functions_list:
            if func.name.startswith(current_string + next_token):
                return True
        return False

    def _valid_token_ids(self, current_string: str) -> list[int]:
        valid_tokens: list[int] = []
        for token, ids in self.vocab_dict.items():
            if self._check_next_token(current_string, token):
                valid_tokens.append(ids)
        return valid_tokens

    def _modify_logits(
        self, logits: list[float], valid_tokens: list[int]) -> list[float]:
        modified_logits: list[float] = []
        for index, log in enumerate(logits):
            if index not in valid_tokens:
                modified_logits.append(float('-inf'))
            else:
                modified_logits.append(log)
        return modified_logits

    def _valid_num_token_ids(self) -> list[int]:
        valid_tokens: list[int] = []
        for token, ids in self.vocab_dict.items():
            if all(c in "0123456789.-" for c in token):
                valid_tokens.append(ids)
        return valid_tokens
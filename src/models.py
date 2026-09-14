from pydantic import BaseModel, TypeAdapter, ValidationError
from typing import TypeVar, Type


T = TypeVar("T", bound=BaseModel)


class Types(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, Types]
    returns: Types


class PromptDefinition(BaseModel):
    prompt: str


def parse_json_list(
        path: str, definiton_type: Type[T]) -> list[T]:
    try:
        with open(path) as f:
            data = f.read()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error with the given path: {e}")
    data_list_adapter = TypeAdapter(list[definiton_type])
    try:
        data_list: list[T] = (
            data_list_adapter.validate_json(data))
    except ValidationError as e:
        raise ValueError(f"Error in definition file: {e}") from e
    return data_list

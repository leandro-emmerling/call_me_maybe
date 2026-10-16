from pydantic import BaseModel, TypeAdapter, ValidationError
from typing import TypeVar, Type


T = TypeVar("T", bound=BaseModel)


class Types(BaseModel):
    """Represents a type definition with a single type attribute."""
    type: str


class FunctionDefinition(BaseModel):
    """Represents a complete function defintion including name,
    description, parameters and return type"""
    name: str
    description: str
    parameters: dict[str, Types]
    returns: Types


class PromptDefinition(BaseModel):
    """Represents a single prompt entry from the input file."""
    prompt: str


def parse_json_list(path: str, cls: Type[T]) -> list[T]:
    """Parse a JSON file and validates its contents against a Pydantic model.

    Args:
        path: path to the JSON file.
        cls: the Pydantic model class to validate against.

    Raises:
        FileNotFoundError: if the file path does not exist.
        ValueError: if the file contains invalid JSON or doesn't match
            the schema.

    Returns:
        A list of validated model instances.
    """
    try:
        with open(path) as f:
            data = f.read()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error with the given path: {e}")
    data_list_adapter = TypeAdapter(list[cls])  # type: ignore[valid-type]
    # mypy TypeVar false positive
    try:
        data_list: list[T] = (
            data_list_adapter.validate_json(data))
    except ValidationError as e:
        raise ValueError(f"Error in definition file: {e}") from e
    return data_list

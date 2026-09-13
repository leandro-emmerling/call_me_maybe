#!/usr/bin/env python3


import json
from pydantic import BaseModel, TypeAdapter


class Type(BaseModel):
    type: str

class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, Type]
    returns: Type


def parse_json_function_list(path: str) -> list[FunctionDefinition]:
    with open(path) as f:
        data = f.read()
    data_list_adapter = TypeAdapter(list[FunctionDefinition])
    data_list: list[FunctionDefinition] = data_list_adapter.validate_json(data)
    return data_list

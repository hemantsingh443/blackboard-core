"""Schema helpers for ecosystem integration wrappers."""

from typing import Any, Dict, List, Type

from pydantic import Field, create_model

from blackboard.protocols import WorkerInput


def _json_type_to_python(prop: Dict[str, Any]) -> Any:
    """Map common JSON Schema types to Python annotations."""
    json_type = prop.get("type", "string")
    if isinstance(json_type, list):
        non_null = [item for item in json_type if item != "null"]
        json_type = non_null[0] if non_null else "string"

    if json_type == "string":
        return str
    if json_type == "integer":
        return int
    if json_type == "number":
        return float
    if json_type == "boolean":
        return bool
    if json_type == "array":
        return List[Any]
    if json_type == "object":
        return Dict[str, Any]
    return Any


def json_schema_to_worker_input(
    schema: Dict[str, Any],
    model_name: str = "IntegrationInput",
) -> Type[WorkerInput]:
    """Create a WorkerInput subclass from a JSON Schema object."""
    properties = schema.get("properties", {}) if schema else {}
    required = set(schema.get("required", [])) if schema else set()

    fields = {}
    for field_name, prop in properties.items():
        annotation = _json_type_to_python(prop)
        default = ... if field_name in required else prop.get("default", None)
        description = prop.get("description")
        fields[field_name] = (
            annotation,
            Field(default, description=description) if description else default,
        )

    return create_model(model_name, __base__=WorkerInput, **fields)

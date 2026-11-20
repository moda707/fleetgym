"""Building blocks shared by all spec models.

* `FrozenModel`: immutable pydantic base model that rejects unknown keys.
* `FrozenMap` and `FrozenJSON`: read-only containers, so a spec is immutable all the way down.
* `InstanceRef`: a value taken from the instance file (``from_instance``).
* `PluginEntry` and `validate_plugin_params`: registry-driven parameter models for plugins
  (constraints, distributions, recourse rules, ...), with error locations preserved.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Annotated, Any, ClassVar, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    GetJsonSchemaHandler,
    PlainSerializer,
    ValidationError,
    WrapSerializer,
    model_serializer,
    model_validator,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, InitErrorDetails

from fleetgym.plugins import PluginError, Registry
from fleetgym.spec.registries import PluginEntry

__all__ = [
    "FROM_INSTANCE",
    "FrozenJSON",
    "FrozenMap",
    "FrozenModel",
    "InstanceRef",
    "NonNegativeOrRef",
    "NumberOrRef",
    "PluginEntry",
    "PositiveOrRef",
    "ReservedParams",
    "deep_freeze",
    "deep_thaw",
    "reraise_nested",
    "validate_plugin_params",
]

FROM_INSTANCE = "from_instance"


class FrozenModel(BaseModel):
    """Immutable pydantic model that rejects unknown keys and validates defaults."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_default=True,
        use_enum_values=True,
        populate_by_name=True,
    )


# --------------------------------------------------------------------------- frozen containers


def deep_freeze(value: Any) -> Any:
    """Return a read-only copy: dicts become mapping proxies, lists become tuples."""
    if isinstance(value, Mapping):
        return MappingProxyType({k: deep_freeze(v) for k, v in value.items()})
    if isinstance(value, list | tuple):
        return tuple(deep_freeze(v) for v in value)
    return value


def deep_thaw(value: Any) -> Any:
    """Inverse of `deep_freeze` for serialization: mappings to dicts, tuples to lists."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {k: deep_thaw(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [deep_thaw(v) for v in value]
    return value


def _freeze_map(value: Mapping[Any, Any]) -> Mapping[Any, Any]:
    return MappingProxyType(dict(value))


def _serialize_map(value: Mapping[Any, Any], handler: Callable[[Any], Any]) -> Any:
    return handler(dict(value))


type FrozenMap[K, V] = Annotated[
    Mapping[K, V], AfterValidator(_freeze_map), WrapSerializer(_serialize_map)
]
"""Read-only mapping. Validates like ``dict[K, V]``; mutation raises ``TypeError``."""

FrozenJSON = Annotated[Any, AfterValidator(deep_freeze), PlainSerializer(deep_thaw)]
"""Free-form JSON-like data (documentation fields, opaque reserved sections), deeply frozen."""


# --------------------------------------------------------------------------- instance references


class InstanceRef(FrozenModel):
    """A value read from the instance file when the spec is bound to an instance.

    YAML forms
    ----------
    ``from_instance``
        Use the field's default instance key (documented per field).
    ``{from_instance: <key>}``
        Use an explicit instance key, for example ``{from_instance: demands}``.
    """

    key: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _parse(cls, data: Any) -> Any:
        if isinstance(data, InstanceRef):
            return data
        if isinstance(data, Mapping) and set(data) <= {"key"}:
            return data  # Python form: InstanceRef() or InstanceRef(key="demands")
        if data == FROM_INSTANCE:
            return {"key": None}
        if isinstance(data, Mapping) and set(data) == {FROM_INSTANCE}:
            key = data[FROM_INSTANCE]
            if not isinstance(key, str) or not key:
                raise ValueError("`from_instance` key must be a non-empty string")
            return {"key": key}
        raise ValueError("expected `from_instance` or `{from_instance: <key>}`")

    @model_serializer
    def _serialize(self) -> str | dict[str, str]:
        return FROM_INSTANCE if self.key is None else {FROM_INSTANCE: self.key}

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        return {
            "anyOf": [
                {"const": FROM_INSTANCE},
                {
                    "type": "object",
                    "properties": {FROM_INSTANCE: {"type": "string", "minLength": 1}},
                    "required": [FROM_INSTANCE],
                    "additionalProperties": False,
                },
            ],
            "description": "Value read from the instance file.",
        }


NumberOrRef = float | InstanceRef
PositiveOrRef = Annotated[float, Field(gt=0)] | InstanceRef
NonNegativeOrRef = Annotated[float, Field(ge=0)] | InstanceRef


# --------------------------------------------------------------------------- plugin parameters


class ReservedParams(FrozenModel):
    """Opaque parameters of a reserved (not yet implemented) feature.

    Any keys are accepted so that planned configs validate; using the feature raises
    `fleetgym.errors.NotInScopeError` when an environment is made.
    """

    model_config = ConfigDict(extra="allow")


def reraise_nested(
    error: ValidationError, title: str, prefix: tuple[str | int, ...] = ()
) -> ValidationError:
    """Rebuild ``error`` so its locations are relative to an enclosing model.

    Pydantic does not merge a `ValidationError` raised inside a validator into the outer error;
    rebuilding it with `ValidationError.from_exception_data` keeps every nested location.
    """
    details: list[InitErrorDetails] = []
    for err in error.errors():
        detail: InitErrorDetails = {
            "type": err["type"],  # pyright: ignore[reportAssignmentType]
            "loc": (*prefix, *err["loc"]),
            "input": err["input"],
        }
        if "ctx" in err:
            detail["ctx"] = err["ctx"]
        details.append(detail)
    return ValidationError.from_exception_data(title, details)


def validate_plugin_params(
    registry: Registry[PluginEntry],
    name: str,
    data: Mapping[str, Any],
    *,
    title: str,
    loc: tuple[str | int, ...] = (),
) -> FrozenModel:
    """Validate ``data`` against the parameter model registered as ``name``.

    Raises
    ------
    ValidationError
        With locations prefixed by ``loc``; an unknown name is reported at ``loc``.
    """
    try:
        entry = registry.get(name)
    except PluginError as exc:
        raise ValidationError.from_exception_data(
            title,
            [
                {
                    "type": "value_error",
                    "loc": loc,
                    "input": name,
                    "ctx": {"error": ValueError(str(exc))},
                }
            ],
        ) from None
    try:
        return entry.params.model_validate(dict(data))
    except ValidationError as exc:
        raise reraise_nested(exc, title, loc) from None


def plugin_variants_schema(
    registry: Registry[PluginEntry],
    discriminator: str,
    handler: GetJsonSchemaHandler,
    *,
    extra_properties: Mapping[str, JsonSchemaValue] | None = None,
) -> JsonSchemaValue:
    """JSON Schema ``oneOf`` with one flat variant per registered plugin."""
    variants: list[JsonSchemaValue] = []
    for name, entry in registry.items():
        params_schema = handler.resolve_ref_schema(handler(entry.params.__pydantic_core_schema__))
        properties = {
            discriminator: {"const": name},
            **dict(params_schema.get("properties", {})),
            **dict(extra_properties or {}),
        }
        variant: JsonSchemaValue = {
            "type": "object",
            "properties": properties,
            "required": [discriminator, *params_schema.get("required", [])],
            "additionalProperties": bool(entry.params.model_config.get("extra") == "allow"),
        }
        suffix = " (reserved: not in this version)" if entry.reserved else ""
        description = entry.description + suffix
        if description:
            variant["description"] = description.strip()
        variants.append(variant)
    return {"oneOf": variants}


StatusLiteral = Literal["stable", "experimental", "planned"]


class FlatPluginModel(FrozenModel):
    """Base for specs written flat in YAML, ``{<discriminator>: <name>, **params}``.

    Subclasses set `discriminator` and `registry`. In Python the model has two fields,
    ``name`` and ``params``; in YAML and JSON it is flat.
    """

    discriminator: ClassVar[str]
    registry: ClassVar[Registry[PluginEntry]]
    extra_keys: ClassVar[tuple[str, ...]] = ()

    name: str
    params: FrozenModel

    @classmethod
    def _preprocess(cls, data: Any) -> Any:
        """Normalize shorthand input before validation (hook for subclasses)."""
        return data

    @model_validator(mode="wrap")
    @classmethod
    def _flatten_in(cls, data: Any, handler: Callable[[Any], Any]) -> Any:
        data = cls._preprocess(data)
        if isinstance(data, Mapping) and cls.discriminator in data:
            name = data[cls.discriminator]
            if not isinstance(name, str):
                raise ValueError(f"`{cls.discriminator}` must be a string")
            rest = {
                k: v for k, v in data.items() if k != cls.discriminator and k not in cls.extra_keys
            }
            params = validate_plugin_params(cls.registry, name, rest, title=cls.__name__)
            return handler({"name": name, "params": params})
        if isinstance(data, Mapping) and "name" not in data:
            raise ValueError(f"missing `{cls.discriminator}`")
        return handler(data)

    @model_serializer(mode="wrap")
    def _flatten_out(self, handler: Callable[[Any], Any]) -> dict[str, Any]:
        return {self.discriminator: self.name, **self.params.model_dump(mode="json")}

    @property
    def reserved(self) -> bool:
        """True if this plugin is reserved (not implemented in this version)."""
        return self.registry.get(self.name).reserved

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        return plugin_variants_schema(cls.registry, cls.discriminator, handler)

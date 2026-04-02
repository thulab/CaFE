"""Application layer for the TS dynamic benchmark backend."""

from __future__ import annotations

import json

from pydantic import BaseModel


def _model_dump(self: BaseModel, *args, **kwargs):
    mode = kwargs.pop("mode", None)
    if mode == "json":
        return json.loads(self.json(*args, **kwargs))
    return self.dict(*args, **kwargs)


@classmethod
def _model_validate(cls, value):
    return cls.parse_obj(value)


def _model_copy(self: BaseModel, *args, **kwargs):
    return self.copy(*args, **kwargs)


if not hasattr(BaseModel, "model_dump"):
    BaseModel.model_dump = _model_dump  # type: ignore[attr-defined]

if not hasattr(BaseModel, "model_validate"):
    BaseModel.model_validate = _model_validate  # type: ignore[attr-defined]

if not hasattr(BaseModel, "model_copy"):
    BaseModel.model_copy = _model_copy  # type: ignore[attr-defined]

from __future__ import annotations

from typing import Any, Callable

try:
    from typing_extensions import TypedDict, Unpack
except Exception:  # pragma: no cover - old environments
    from typing import TypedDict  # type: ignore

    class _Unpack:
        def __class_getitem__(cls, item):
            return item

    Unpack = _Unpack  # type: ignore


class TransformersKwargs(TypedDict, total=False):
    """Small fallback for Transformers versions that moved this annotation."""


def identity_decorator(obj: Callable | None = None, **_kwargs: Any):
    def decorate(value):
        return value

    if callable(obj):
        return obj
    return decorate


try:
    from transformers.utils.generic import check_model_inputs as _check_model_inputs
except Exception:
    _check_model_inputs = identity_decorator


# The generated upstream file has custom generation-only arguments that some
# Transformers releases reject during docstring validation. Runtime behavior does
# not depend on that decorator, so keep it inert across versions.
auto_docstring = identity_decorator
check_model_inputs = _check_model_inputs

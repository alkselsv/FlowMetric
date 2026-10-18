"""Compute-device selection shared by command-line entry points."""

from __future__ import annotations

import re

import torch


SUPPORTED_DEVICE_FORMS = "auto, cpu, mps, cuda, cuda:N or N"


def resolve_device(requested: str | None = "auto") -> str:
    """Return a validated canonical device accepted by PyTorch and Ultralytics.

    ``auto`` prefers CUDA, then Apple MPS, and finally CPU. Explicit requests
    fail early instead of silently falling back to a different device.
    """
    if requested is None:
        value = "auto"
    elif not isinstance(requested, str):
        raise TypeError("device must be a string or None")
    else:
        value = requested.strip().lower()

    if not value:
        raise ValueError(f"device must be one of: {SUPPORTED_DEVICE_FORMS}")
    if value == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if value == "cpu":
        return value
    if value in {"mps", "mps:0"}:
        if not torch.backends.mps.is_built():
            raise ValueError("MPS was requested, but this PyTorch build has no MPS support")
        if not torch.backends.mps.is_available():
            raise ValueError(
                "MPS was requested, but it is unavailable to this process. "
                "Run outside a GPU-restricted sandbox and check macOS/PyTorch support."
            )
        return "mps"

    match = re.fullmatch(r"(?:cuda:)?(\d+)", value)
    if value == "cuda":
        index = 0
    elif match:
        index = int(match.group(1))
    else:
        raise ValueError(f"device must be one of: {SUPPORTED_DEVICE_FORMS}; got {requested!r}")
    if not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but torch.cuda.is_available() is false")
    count = torch.cuda.device_count()
    if index >= count:
        raise ValueError(f"CUDA device {index} was requested, but only {count} device(s) are visible")
    return f"cuda:{index}"

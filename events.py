"""Compatibility alias for :mod:`flowmetric.events`."""

import sys

from flowmetric import events as _implementation

sys.modules[__name__] = _implementation

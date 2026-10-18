"""Compatibility alias for :mod:`flowmetric.identity.gallery`."""

import sys

from flowmetric.identity import gallery as _implementation

sys.modules[__name__] = _implementation

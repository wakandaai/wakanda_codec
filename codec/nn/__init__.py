"""Neural-network utilities package for codec.

Expose common submodules at package level so callers can use
`codec.nn.loss`, `codec.nn.layers`, and `codec.nn.quantize`.
"""

# Re-export submodules for convenience and to match expected API.
from codec.nn import loss
from codec.nn import layers
from codec.nn import quantize

__all__ = ["loss", "layers", "quantize"]


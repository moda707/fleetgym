"""Neural network base classes and built-in architectures (layer L6, extra `sb3`).

Importing this package needs PyTorch (``pip install fleetgym[sb3]``). `MaskableGraphPolicy` also
needs ``sb3-contrib`` and is imported from `fleetgym.nn.policies`.
"""

from fleetgym.nn.adapters import GraphBatch, to_graph_batch
from fleetgym.nn.base import Decoder, Embeddings, Encoder, FleetGymModule, masked_mean
from fleetgym.nn.decoders import PointerDecoder
from fleetgym.nn.encoders import AttentionModelEncoder, GATEncoder, MLPEncoder
from fleetgym.nn.registry import MODULES, get_module, register

__all__ = [
    "MODULES",
    "AttentionModelEncoder",
    "Decoder",
    "Embeddings",
    "Encoder",
    "FleetGymModule",
    "GATEncoder",
    "GraphBatch",
    "MLPEncoder",
    "PointerDecoder",
    "get_module",
    "masked_mean",
    "register",
    "to_graph_batch",
]

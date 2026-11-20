"""Checks for custom neural network modules (design §20, customization contract).

`check_module(encoder, decoder)` runs the pair on synthetic `GraphBatch` inputs and checks:

* shapes: node embeddings ``(B, N, D)``, context ``(B, D)``, logits ``(B, N + extra)``;
* masking: logits are ``-inf`` exactly at masked actions (finite elsewhere);
* permutation equivariance over customers (for modules declaring ``equivariant = True``): permuting
  customer slots permutes node embeddings and node logits and leaves the context unchanged;
* batch independence: a sample's outputs do not depend on the other samples in the batch;
* padding invariance: adding empty padding slots changes no output of the real nodes (skipped for
  modules with ``fixed_size = True``).

Requires PyTorch (extra ``sb3``).
"""

from __future__ import annotations

import torch

from fleetgym.errors import FleetGymError
from fleetgym.nn.adapters import GraphBatch
from fleetgym.nn.base import Decoder, Embeddings, Encoder
from fleetgym.nn.decoders import PointerDecoder

__all__ = ["ModuleCheckError", "check_module", "synthetic_batch"]


class ModuleCheckError(FleetGymError):
    """A module breaks the FleetGym module contract."""


def synthetic_batch(
    batch: int,
    n_nodes: int,
    *,
    node_features: int = 10,
    vehicle_features: int = 6,
    global_features: int = 4,
    n_extra: int = 1,
    padding: int = 0,
    seed: int = 0,
) -> GraphBatch:
    """Random graphs with ``n_nodes`` real nodes, then ``padding`` empty slots."""
    gen = torch.Generator().manual_seed(seed)
    n = n_nodes + padding
    nodes = torch.zeros(batch, n, node_features)
    nodes[:, :n_nodes] = torch.rand(batch, n_nodes, node_features, generator=gen)
    valid = torch.zeros(batch, n, dtype=torch.bool)
    valid[:, :n_nodes] = True
    mask = torch.zeros(batch, n + n_extra, dtype=torch.bool)
    mask[:, :n_nodes] = torch.rand(batch, n_nodes, generator=gen) > 0.4
    mask[:, n:] = True
    return GraphBatch(
        nodes=nodes,
        node_valid=valid,
        vehicle=torch.rand(batch, vehicle_features, generator=gen),
        globals=torch.rand(batch, global_features, generator=gen),
        action_mask=mask,
    )


def _run(encoder: Encoder, decoder: Decoder, g: GraphBatch) -> tuple[Embeddings, torch.Tensor]:
    with torch.no_grad():
        emb = encoder(g)
        return emb, decoder(emb, g)


def _close(a: torch.Tensor, b: torch.Tensor, atol: float) -> bool:
    finite = torch.isfinite(a) & torch.isfinite(b)
    same_inf = (torch.isinf(a) == torch.isinf(b)).all()
    return bool(same_inf) and bool(torch.allclose(a[finite], b[finite], atol=atol))


def check_module(
    encoder: Encoder,
    decoder: Decoder | None = None,
    *,
    n_nodes: int = 6,
    batch: int = 3,
    n_extra: int = 1,
    seed: int = 0,
    atol: float = 1e-5,
) -> None:
    """Check an encoder (and decoder) against the module contract.

    The encoder must take inputs of 10 node, 6 vehicle and 4 global features (the FleetGym
    observation layout). The default decoder is a `PointerDecoder`.

    Raises
    ------
    ModuleCheckError
        Listing every failed check.
    """
    encoder.eval()
    dim = encoder.embed_dim
    head: Decoder = decoder if decoder is not None else PointerDecoder(dim, n_extra=n_extra)
    head.eval()
    failures: list[str] = []
    g = synthetic_batch(batch, n_nodes, n_extra=n_extra, seed=seed)
    emb, logits = _run(encoder, head, g)

    if emb.node.shape != (batch, n_nodes, dim):
        failures.append(f"node embeddings have shape {tuple(emb.node.shape)}")
    if emb.context.shape != (batch, dim):
        failures.append(f"context has shape {tuple(emb.context.shape)}")
    if logits.shape != (batch, n_nodes + n_extra):
        failures.append(f"logits have shape {tuple(logits.shape)}")
    elif not (
        torch.isneginf(logits[~g.action_mask]).all() and torch.isfinite(logits[g.action_mask]).all()
    ):
        failures.append("logits must be -inf exactly at masked actions")

    if encoder.equivariant and head.equivariant:
        perm = torch.cat([torch.tensor([0]), 1 + torch.randperm(n_nodes - 1)])
        permuted = GraphBatch(
            nodes=g.nodes[:, perm],
            node_valid=g.node_valid[:, perm],
            vehicle=g.vehicle,
            globals=g.globals,
            action_mask=torch.cat([g.action_mask[:, perm], g.action_mask[:, n_nodes:]], dim=1),
        )
        p_emb, p_logits = _run(encoder, head, permuted)
        if not torch.allclose(p_emb.node, emb.node[:, perm], atol=atol):
            failures.append("node embeddings are not permutation equivariant over customers")
        if not torch.allclose(p_emb.context, emb.context, atol=atol):
            failures.append("the context changes when customers are permuted")
        expected = torch.cat([logits[:, perm], logits[:, n_nodes:]], dim=1)
        if not _close(p_logits, expected, atol):
            failures.append("logits are not permutation equivariant over customers")

    single_emb, single_logits = _run(
        encoder,
        head,
        GraphBatch(
            nodes=g.nodes[:1],
            node_valid=g.node_valid[:1],
            vehicle=g.vehicle[:1],
            globals=g.globals[:1],
            action_mask=g.action_mask[:1],
        ),
    )
    if not (
        torch.allclose(single_emb.context, emb.context[:1], atol=atol)
        and _close(single_logits, logits[:1], atol)
    ):
        failures.append("outputs depend on the other samples of the batch")

    if encoder.fixed_size:
        if failures:
            raise ModuleCheckError(f"{type(encoder).__name__}: " + "; ".join(failures))
        return
    pad = 3
    padded = synthetic_batch(batch, n_nodes, n_extra=n_extra, padding=pad, seed=seed)
    pad_emb, pad_logits = _run(encoder, head, padded)
    if not torch.allclose(pad_emb.node[:, :n_nodes], emb.node, atol=atol):
        failures.append("padding slots change the node embeddings")
    if not torch.allclose(pad_emb.context, emb.context, atol=atol):
        failures.append("padding slots change the context")
    pad_real = torch.cat([pad_logits[:, :n_nodes], pad_logits[:, n_nodes + pad :]], dim=1)
    if not _close(pad_real, logits, atol):
        failures.append("padding slots change the logits of real actions")

    if failures:
        raise ModuleCheckError(f"{type(encoder).__name__}: " + "; ".join(failures))

"""Neural network modules: the module contract (check_module) and the SB3 graph policy."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sb3_contrib")

from fleetgym.nn import (  # noqa: E402
    AttentionModelEncoder,
    Embeddings,
    Encoder,
    GATEncoder,
    MLPEncoder,
    PointerDecoder,
    get_module,
    register,
)
from fleetgym.nn.adapters import GraphBatch  # noqa: E402
from fleetgym.testing.module_checker import ModuleCheckError, check_module, synthetic_batch  # noqa: E402
from tests.families import make_env  # noqa: E402

pytestmark = pytest.mark.extras_sb3


@pytest.mark.parametrize(
    "encoder",
    [
        lambda: GATEncoder(10, 6, 4, 32, n_layers=2, heads=4),
        lambda: AttentionModelEncoder(10, 6, 4, 32, n_layers=2, heads=4),
        lambda: MLPEncoder(10, 6, 4, 32, max_nodes=6),
    ],
    ids=["gat", "attention", "mlp"],
)
def test_built_in_modules_pass_the_contract(encoder):
    check_module(encoder())


class LeakyEncoder(Encoder):
    """Breaks padding invariance: pools over every slot, padding included."""

    def forward(self, g: GraphBatch) -> Embeddings:
        h = self.node_in(g.nodes)
        context = torch.tanh(
            self.context_in(torch.cat([h.mean(dim=1), g.vehicle, g.globals], dim=-1))
        )
        return Embeddings(node=h * g.node_valid.unsqueeze(-1), context=context)


class UnmaskedDecoder(PointerDecoder):
    """Breaks masking: returns raw logits."""

    def forward(self, emb, g):
        return self.logits(emb)


def test_broken_modules_are_caught():
    with pytest.raises(ModuleCheckError, match="padding"):
        check_module(LeakyEncoder(10, 6, 4, 16))
    with pytest.raises(ModuleCheckError, match="-inf"):
        check_module(GATEncoder(10, 6, 4, 16), UnmaskedDecoder(16, n_extra=1))


def test_registry_and_batch_helpers():
    assert get_module("gat") is GATEncoder
    register("tests.leaky")(LeakyEncoder)
    assert get_module("tests.leaky") is LeakyEncoder
    from fleetgym.nn.registry import MODULES

    MODULES.unregister("tests.leaky")
    g = synthetic_batch(2, 4, padding=2)
    assert g.n_nodes == 6 and not g.node_valid[:, 4:].any()


def test_graph_policy_acts_validly():
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from fleetgym.agents.sb3 import SB3Policy
    from fleetgym.baselines import run_episode
    from fleetgym.nn.policies import MaskableGraphPolicy

    env = make_env("VRPTW-v1")
    for encoder in ("gat", "attention", "mlp"):
        model = MaskablePPO(
            MaskableGraphPolicy,
            DummyVecEnv([lambda: make_env("VRPTW-v1")]),
            n_steps=32,
            batch_size=16,
            seed=0,
            policy_kwargs={"features_extractor_kwargs": {"encoder": encoder, "embed_dim": 16}},
        )
        model.learn(64)
        episode = run_episode(env, SB3Policy(model), seed=3)  # masked actions would raise
        assert episode.steps > 0

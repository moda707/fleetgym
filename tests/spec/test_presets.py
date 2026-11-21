"""Paper presets (ADR 0023): a model, the paper's grid, and where the instances come from."""

from __future__ import annotations

import numpy as np
import pytest

import fleetgym as fg
from fleetgym.data import Instance, write_instance_set
from fleetgym.errors import ConfigError
from fleetgym.testing.validator import validate

GRID_SIZES = {
    "dastpak2022-goodson": 27,
    "dastpak2022-vcsd": 15,
    "florio2023-vrpsd": 4,
    "kool2019-cvrp": 3,
    "solomon1987-vrptw": 56,
    "ulmer2016-vrpssr": 18,
    "vera2019-hetfleet": 4,
}


def test_every_shipped_preset_loads_and_cites_its_paper():
    assert set(fg.preset_names()) == set(GRID_SIZES)
    for name, size in GRID_SIZES.items():
        preset = fg.load_preset(name)
        assert len(preset.points()) == size
        assert preset.reference.get("key")
        assert preset.point() == {**preset.default}  # the defaults are on the grid


def test_grid_errors_name_the_preset_and_the_grid():
    preset = fg.load_preset("dastpak2022-vcsd")
    with pytest.raises(ConfigError, match="no dimension"):
        preset.point(colour="red")
    with pytest.raises(ConfigError, match=r"capacity=60 is not on the paper's grid \[25, 50, 75\]"):
        preset.point(capacity=60)
    with pytest.raises(ConfigError, match="pass instances=<folder>"):
        preset.spec()
    with pytest.raises(ConfigError, match="unknown preset"):
        fg.load_preset("nobody2099-vrp")


def kool_like(n: int, seed: int) -> Instance:
    """A test-side stand-in for the example's generation function."""
    rng = np.random.default_rng(seed)
    return Instance(
        name=f"kool-n{n}-{seed}",
        source="tests",
        demands=np.concatenate([[0.0], rng.integers(1, 10, n).astype(float)]),
        coords=rng.uniform(0, 1, (n + 1, 2)),
        vehicle_types=({"capacity": {20: 30, 50: 40, 100: 50}[n]},),
    )


def test_a_folder_preset_runs_on_the_instances_the_user_wrote(tmp_path):
    write_instance_set([kool_like(20, s) for s in range(4)], tmp_path / "n20", name="kool n20")
    env = fg.make_preset("kool2019-cvrp", instances=tmp_path, n=20)
    assert isinstance(env, fg.FleetGymEnv)
    seen = set()
    for seed in range(8):
        env.reset(seed=seed)
        seen.add(env.base_instance.name)
        while not env.simulator.state.terminated:
            env.step(int(np.flatnonzero(env.action_masks())[0]))
        assert validate(env.event_log, env.problem_spec, env.base_instance).ok
    assert len(seen) > 1
    with pytest.raises(ConfigError, match="no instances for"):
        fg.make_preset("kool2019-cvrp", instances=tmp_path, n=50)  # not written


def test_a_multi_agent_preset_is_a_pettingzoo_environment(tmp_path):
    rng = np.random.default_rng(0)
    vrp10 = Instance(
        name="vrp10",
        source="tests",
        demands=np.concatenate([[0.0], rng.integers(1, 10, 10).astype(float)]),
        coords=rng.uniform(0, 1, (11, 2)),
        vehicle_types=tuple({"capacity": c, "count": 1} for c in (10, 15, 20)),
    )
    write_instance_set([vrp10], tmp_path / "vrp10", name="vrp10")
    env = fg.make_preset("vera2019-hetfleet", instances=tmp_path, size="vrp10")
    assert isinstance(env, fg.FleetGymAECEnv)
    assert env.possible_agents == ["vehicle_0", "vehicle_1", "vehicle_2"]


@pytest.mark.network
def test_benchmark_presets_download_their_instances():
    env = fg.make_preset("solomon1987-vrptw", instance="c101")
    assert isinstance(env, fg.FleetGymEnv)
    env.reset(seed=0)
    assert env.base_instance.name.lower() == "c101"
    env = fg.make_preset("florio2023-vrpsd", instance="X-n110-k13")
    assert isinstance(env, fg.FleetGymEnv)
    env.reset(seed=0)
    assert env.scenario is not None and env.base_instance.n_customers == 109

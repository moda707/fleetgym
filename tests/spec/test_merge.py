import pytest

from fleetgym.spec.merge import drop_nulls, merge_configs, merge_maps


@pytest.mark.parametrize(
    ("parent", "child", "expected"),
    [
        # maps merge deeply
        ({"a": {"x": 1, "y": 2}}, {"a": {"y": 3, "z": 4}}, {"a": {"x": 1, "y": 3, "z": 4}}),
        # scalars and lists replace
        ({"a": 1, "l": [1, 2]}, {"a": 2, "l": [3]}, {"a": 2, "l": [3]}),
        # constraints merge by name
        (
            {"constraints": {"capacity": {}, "time_windows": {"soft": False}}},
            {"constraints": {"time_windows": {"soft": True}, "route_duration": {"max": 5}}},
            {
                "constraints": {
                    "capacity": {},
                    "time_windows": {"soft": True},
                    "route_duration": {"max": 5},
                }
            },
        ),
        # null deletes an inherited key
        (
            {"constraints": {"capacity": {}, "x": {}}},
            {"constraints": {"x": None}},
            {"constraints": {"capacity": {}}},
        ),
        ({"a": 1}, {"a": None}, {}),
        # null for a key that is not inherited is simply dropped
        ({}, {"a": None}, {}),
        # a discriminator change replaces the subtree
        (
            {"u": {"dist": "gamma", "shape": 2, "mean": 3, "revealed": "on_arrival"}},
            {"u": {"dist": "poisson", "mean": 4}},
            {"u": {"dist": "poisson", "mean": 4}},
        ),
        # same discriminator merges
        (
            {"u": {"dist": "gamma", "shape": 2, "mean": 3}},
            {"u": {"dist": "gamma", "mean": 5}},
            {"u": {"dist": "gamma", "shape": 2, "mean": 5}},
        ),
        # child map without the discriminator merges into the parent's
        (
            {"n": {"type": "euclidean", "rounding": "none"}},
            {"n": {"speed": 2}},
            {"n": {"type": "euclidean", "rounding": "none", "speed": 2}},
        ),
        # replaced subtrees lose their nulls
        (
            {"u": {"dist": "a"}},
            {"u": {"dist": "b", "p": None, "q": 1}},
            {"u": {"dist": "b", "q": 1}},
        ),
        # map replaces scalar and scalar replaces map
        ({"r": "return_to_depot"}, {"r": {"rule": "split"}}, {"r": {"rule": "split"}}),
        ({"r": {"rule": "split"}}, {"r": "outsource"}, {"r": "outsource"}),
    ],
)
def test_merge_maps(parent, child, expected):
    assert merge_maps(parent, child) == expected


def test_merge_does_not_mutate_inputs():
    parent = {"a": {"x": 1}}
    child = {"a": {"y": 2}}
    merge_maps(parent, child)
    assert parent == {"a": {"x": 1}}
    assert child == {"a": {"y": 2}}


def test_objective_is_replaced_not_merged():
    parent = {"objective": {"lexicographic": ["vehicles", "distance"]}}
    child = {"objective": {"distance": 1.0, "rejected": 50.0}}
    assert merge_configs(parent, child) == {"objective": {"distance": 1.0, "rejected": 50.0}}


def test_meta_and_id_are_not_inherited():
    parent = {
        "id": "A-v0",
        "meta": {"name": "A", "expected": {"x": 1}},
        "fleet": {"sizing": "fixed"},
    }
    assert merge_configs(parent, {}) == {"fleet": {"sizing": "fixed"}}
    assert merge_configs(parent, {"meta": {"name": "B"}}) == {
        "fleet": {"sizing": "fixed"},
        "meta": {"name": "B"},
    }


def test_drop_nulls_keeps_lists():
    assert drop_nulls({"a": None, "b": {"c": None, "d": 1}, "l": [None, 1]}) == {
        "b": {"d": 1},
        "l": [None, 1],
    }

"""Hypothesis: random valid specs round-trip through YAML with the same hash."""

from hypothesis import given
from hypothesis import strategies as st

from fleetgym.spec import load_spec
from fleetgym.spec.models import ProblemSpec
from fleetgym.spec.yaml_io import dump_yaml, load_yaml

finite = st.floats(min_value=0.001, max_value=1e6, allow_nan=False, allow_infinity=False)
ref_or_number = st.one_of(finite, st.just("from_instance"))

distributions = st.one_of(
    st.builds(lambda m: {"dist": "poisson", "mean": m}, ref_or_number),
    st.builds(lambda s, m: {"dist": "gamma", "shape": s, "mean": m}, finite, ref_or_number),
    st.builds(lambda lo, w: {"dist": "uniform", "low": lo, "high": lo + w}, finite, finite),
    st.builds(lambda v: {"dist": "fixed", "value": v}, ref_or_number),
    st.builds(lambda p: {"dist": "bernoulli", "p": p}, st.floats(0, 1)),
)
revealed = st.sampled_from(["a_priori", "on_request", "on_arrival", "on_completion"])
constraints = st.fixed_dictionaries(
    {},
    optional={
        "capacity": st.just({}),
        "time_windows": st.fixed_dictionaries(
            {},
            optional={"soft": st.booleans(), "waiting": st.sampled_from(["allowed", "forbidden"])},
        ),
        "route_duration": st.builds(
            lambda m: {"max": m}, st.one_of(finite, st.just("depot_due_time"))
        ),
    },
)
objectives = st.one_of(
    st.dictionaries(
        st.sampled_from(["distance", "time", "vehicles", "rejected"]), finite, min_size=1
    ),
    st.lists(
        st.sampled_from(["vehicles", "distance", "time"]), min_size=1, max_size=3, unique=True
    ).map(lambda lex: {"lexicographic": lex}),
)
fleets = st.one_of(
    st.builds(
        lambda c: {"sizing": "unlimited", "types": [{"name": "t", "capacity": c}]}, ref_or_number
    ),
    st.builds(
        lambda c, n, s: {"sizing": s, "types": [{"name": "t", "capacity": c, "count": n}]},
        ref_or_number,
        st.one_of(st.integers(1, 50), st.just("from_instance")),
        st.sampled_from(["fixed", "limited"]),
    ),
)
specs = st.fixed_dictionaries(
    {"fleet": fleets, "objective": objectives, "constraints": constraints},
    optional={
        "uncertainty": st.dictionaries(
            st.sampled_from(["demand", "service_time"]),
            st.builds(lambda d, r: {**d, "revealed": r}, distributions, revealed),
            max_size=3,
        ),
        "network": st.sampled_from(
            [
                {"type": "euclidean", "rounding": "nearest_int"},
                {"type": "euclidean"},
                {"type": "matrix"},
            ]
        ),
        "decisions": st.sampled_from(
            [
                {"interface": "next_node"},
                {"epochs": "sequential"},
                {"epochs": "periodic", "period": 5.0},
            ]
        ),
        "recourse": st.sampled_from(
            [
                {},
                {"capacity_failure": "return_to_depot"},
                {"capacity_failure": {"rule": "outsource", "per_unit": 1.5}},
                {"capacity_failure": {"rule": "agent_decides", "options": ["return_to_depot"]}},
            ]
        ),
    },
)


@given(specs)
def test_random_specs_validate_and_round_trip(data):
    spec = ProblemSpec.model_validate(data)
    again = load_spec(load_yaml(dump_yaml(spec.to_dict())))
    assert again == spec
    assert again.content_hash() == spec.content_hash()
    assert ProblemSpec.model_validate(data).content_hash() == spec.content_hash()

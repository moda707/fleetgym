# API reference

## Environments

```{eval-rst}
.. autofunction:: fleetgym.make
.. autofunction:: fleetgym.make_multi
.. autoclass:: fleetgym.FleetGymEnv
   :members: reset, step, action_masks, get_state, set_state, clone, fork, scenario, epoch, base_instance, render
.. autoclass:: fleetgym.FleetGymAECEnv
   :members: reset, step, observe, state
.. autoclass:: fleetgym.FleetGymParallelEnv
   :members: reset, step, state
```

## Specs

```{eval-rst}
.. autofunction:: fleetgym.load_spec
.. autoclass:: fleetgym.ProblemSpec
   :members: content_hash, to_yaml, reserved_features
```

## Extending

```{eval-rst}
.. autofunction:: fleetgym.register_distribution
.. autoclass:: fleetgym.Distribution
   :members: sample, mean, quantile, support
.. autofunction:: fleetgym.register_constraint
.. autoclass:: fleetgym.Constraint
   :members: init_state, mask, check, update, penalty
.. autofunction:: fleetgym.register_recourse
.. autoclass:: fleetgym.RecourseRule
   :members: serve_first, on_failure, after_service, options
```

## Uncertainty and scenarios

```{eval-rst}
.. autoclass:: fleetgym.Scenario
   :members: write, read, check, id
.. automodule:: fleetgym.uncertainty.dynamism
   :members:
```

## Baselines and evaluation

```{eval-rst}
.. automodule:: fleetgym.baselines.protocol
   :members: Policy, run_episode, Episode
.. autoclass:: fleetgym.baselines.NearestNeighbor
.. autoclass:: fleetgym.baselines.CheapestInsertion
.. autoclass:: fleetgym.baselines.Rollout
.. autoclass:: fleetgym.baselines.PerfectInformation
.. automodule:: fleetgym.eval
   :members: evaluate, scenario_seeds, compare, report_card, mean_ci, cvar
```

## Testing tools

```{eval-rst}
.. autofunction:: fleetgym.testing.validator.validate
.. autofunction:: fleetgym.testing.non_anticipativity.check_non_anticipativity
.. autofunction:: fleetgym.testing.non_anticipativity.check_policy_non_anticipative
```

# 0018: Training stack and neural network modules

Status: accepted (2026-11-20). Adopts plan Q31 (recommendations).

## Neural network modules (`fleetgym.nn`, extra `sb3`)
- `to_graph_batch` turns a padded observation dictionary into a dense `GraphBatch` and normalizes
  node features per sample (each feature divided by ``1 +`` its largest absolute value over valid
  nodes), which is permutation equivariant and padding invariant.
- Base classes: `Encoder` (`GraphBatch -> Embeddings(node, context)`, with `embed_nodes` and `pack`
  helpers), `Decoder` (masked logits, `-inf` at masked actions), `FleetGymModule` with
  `equivariant` and `fixed_size` flags; `@register("name")` and `get_module`.
- Built-ins: `MLPEncoder` (flat; not equivariant, fixed size), `GATEncoder` (dense, mask-aware
  multi-head graph attention in pure torch, Q31a; a torch-geometric version is future work),
  `AttentionModelEncoder` (transformer encoder with padding masks), `PointerDecoder` (Kool et al.
  2019 clipped pointer logits; extra actions such as wait from the context).
- `MaskableGraphPolicy` (Q31b, name per Q31c): a sb3-contrib `MaskableActorCriticPolicy` whose
  features extractor runs the encoder and returns the flattened node embeddings and context, whose
  action network is a pointer head and whose value network reads the context. SB3's own rollout,
  masking and saving code is unchanged.
- `fleetgym.testing.check_module(encoder, decoder)`: shapes, `-inf` exactly at masked actions,
  permutation equivariance over customers, batch independence, padding invariance.

## Agents (`fleetgym.agents`, extra `sb3`)
- `MaskableDQN`: masked Q-network (`-inf` at masked actions, read from `obs["action_mask"]`), so
  greedy selection and the target's max respect the masks; epsilon-greedy exploration and warm-up
  sample valid actions only; a fully masked next state bootstraps 0. NumPy's global generator is used
  for exploration, as in SB3, so `seed=` keeps training reproducible.
- `make_sb3_vec`, `sb3_defaults` (YAML files per algorithm; untuned starting points, documented as
  such), `policy_settings`, `SB3Policy` (a trained model as a `Policy`), callbacks `ScenarioEval`
  and `CostComponentLogger`.
- CLI: `fleetgym train <env> --algo --policy --steps --seed` writes `model.zip` and `run.json`;
  `fleetgym eval <env> --model DIR | --baseline NAME` evaluates on fixed test seeds, validates every
  episode and writes a report card.

## Layers
`agents` may import `baselines` (same layer; trained models implement the `Policy` protocol).
Training callbacks use `run_episode`, not the L7 harness.

## CI
The lint job installs the `sb3`, `solvers` and `viz` extras to type-check optional modules; a new
job runs the tests marked `extras_*` with those extras. Default test jobs skip modules that need a
missing extra.

## Verification
`check_module` passes for every built-in module and catches a padding leak and unmasked logits;
MaskableDQN never selects masked actions (greedy, exploration and warm-up) and its targets ignore
masked actions; the graph policy acts validly with every encoder; the CLI trains and evaluates both
algorithms; MaskablePPO learns the optimal route of a fixed 4-customer instance (slow test).

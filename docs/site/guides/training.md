# Training agents

## With Stable-Baselines3

FleetGym environments work with SB3 and sb3-contrib out of the box (dictionary observations,
`action_masks()` for MaskablePPO). `MaskableGraphPolicy` keeps one embedding per node and points at
nodes with an attention decoder; `MaskableDQN` masks Q-values in selection and targets.

```{literalinclude} ../../../examples/06_train_agent/train_agent.py
:language: python
```

From the command line:

```bash
fleetgym train CVRP-v0 --algo maskable_ppo --policy gat --steps 200000 --out runs/cvrp
fleetgym eval CVRP-v0 --model runs/cvrp --episodes 64 --reference nearest
```

## Your own network

Write an encoder (`GraphBatch -> Embeddings`) or a decoder (`Embeddings, GraphBatch -> logits`),
register it, and check it against the module contract: shapes, `-inf` at masked actions, permutation
equivariance over customers, batch independence and padding invariance.

```python
import torch
from torch import nn

import fleetgym.nn as fnn
from fleetgym.testing.module_checker import check_module


@fnn.register("my_encoder")
class MyEncoder(fnn.Encoder):
    def __init__(self, node_features, vehicle_features, global_features, embed_dim=64, **_):
        super().__init__(node_features, vehicle_features, global_features, embed_dim)
        self.mix = nn.Linear(embed_dim, embed_dim)

    def forward(self, g):
        h = self.embed_nodes(g)                       # zero at padding slots
        h = h + torch.relu(self.mix(h)) * g.node_valid.unsqueeze(-1)
        return self.pack(h, g)                        # node embeddings and a context


check_module(MyEncoder(10, 6, 4, 32))                 # raises ModuleCheckError if broken
```

Then use it by name: `policy_kwargs=dict(features_extractor_kwargs=dict(encoder="my_encoder"))`.

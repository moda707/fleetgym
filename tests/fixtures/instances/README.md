# Test fixtures

Hand-written by the FleetGym authors for unit tests (no third-party data):

- `tiny.vrp`: 5-node CVRP, `EUC_2D`; node 5 is exactly 2.5 from the depot to test tie rounding.
- `tiny_explicit.vrp`: 3-node asymmetric `EXPLICIT` matrix with scalar `SERVICE_TIME` and `VEHICLES`.
- `tiny_depot2.vrp`: depot is not node 1 (unsupported layout, must be rejected).
- `tiny_solomon.txt`: 4-node instance in Solomon format.
- `tiny.sol`: VRPLIB-style solution file.

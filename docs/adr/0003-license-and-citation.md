# 0003: MIT license, citation through CITATION.cff

Status: accepted (2026-11-20)

## Context
The user wants people who use FleetGym in research to cite it. Design §25 lists MIT or Apache-2.0.

## Decision
MIT, the license used by the surrounding ecosystem (Gymnasium, PettingZoo, Stable-Baselines3, RL4CO,
PyVRP, TorchRL). Citation is requested, not required: `CITATION.cff` (GitHub shows "Cite this
repository"), a "please cite" section in the README and docs, and a Zenodo DOI per release. Once the
software paper is out, `CITATION.cff` gets a `preferred-citation` entry pointing to it.

## Alternatives
- Apache-2.0: adds an explicit patent grant and a NOTICE file that redistributors must keep. Useful for
  industrial contributors, little benefit for a research library.
- A license with a citation clause: not OSI-approved, discourages adoption, and is generally not
  enforceable for academic citation.

## Consequences
The license can still be changed before the first public release; after that, a change needs every
contributor's agreement.

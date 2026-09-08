# Residual surface-code router compiler

Precommitted confirmatory result, 8 September 2026.

## Outcome

The compiled residual policy beats the 25-way calibration-selected static MWPM
benchmark at both declared distances:

| distance | static LER | compiled LER | paired change (95% CI, pp) |
| ---: | ---: | ---: | ---: |
| 3 | 2.0813% | 2.0214% | -0.0599 [-0.0883, -0.0315] |
| 5 | 1.8822% | 1.8394% | -0.0427 [-0.0812, -0.0043] |

This meets the precommitted benchmark-win condition. It is not yet the selected
architecture: the compiler evaluated the residual head on the same validation records
used to fit that head. At distance 5 it chose residual routing even though the simpler
causal posterior reaches 1.7204% on final test. The residual is worse than that prior
adaptive baseline by +0.1190 pp, 95% CI [+0.0975, +0.1406].

## Other conditions

- **Distance-3 frontier: NO-GO.** Residual versus the prior aggregate router is
  -0.0046 pp with interval [-0.0242, +0.0151]; it is unresolved. Distance 5 violates
  the non-inferiority clause.
- **Oracle recovery: NO-GO.** Recovery is 42.7% at distance 3 and 22.8% at distance 5.
- **Memory value: GO.** Detailed causal posterior routing beats detailed memoryless
  routing at both distances.
- **Stationary safety: NO-GO.** Three of four controls exceed the +0.05 pp upper-bound
  margin; the worst is +0.2104 pp.

The result establishes that constrained residual routing can beat a strong static
benchmark without replacing the decoder. It also exposes selection overfit: fitting
and selecting the residual on one validation set is insufficient. The next compiler
uses disjoint action-training and policy-selection streams and adds each endpoint
decoder's matching energy to the residual inputs. Those energies are graph-native
signals of how costly each explanation is, rather than a larger generic network.

Protocol: [surface-residual-router-protocol.md](surface-residual-router-protocol.md).
Artifact: `results/surface_residual_router.json`.

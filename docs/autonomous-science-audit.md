# Independent audit of the autonomous campaign

The audit verdict is that the current streaming work is a useful causal-estimation
sanity check, not yet quantum control, architectural superiority, or deployment.

Critical corrections:

1. The reported squared error is a delayed-state Brier score. No action changes the
   dynamics and no physical terminal fidelity is propagated.
2. A stationary-prior baseline was absent. For the first generator its expected risk
   is 0.0988, slightly below the fitted forecast at the longest delays.
3. The increasing advantage over a stale current belief follows two-state Bayes
   decision theory and is not architectural novelty.
4. The eight confirmation cells are fixed conditions; intervals across them are
   descriptive, not population-generalization intervals.
5. The causal spline head uses privileged future-state labels and is properly called
   KAN-inspired, not a full KAN deployment result.

The required next ladder is: stationary prior; tuned robust EWMA; multi-start Gaussian
HMM; clipped and contaminated-emission HMMs; linear, spline, and tiny-MLP slow
calibrators; and oracle parameter controls. Retain every parameter fit, convergence
status, prediction, and batch-one latency sample.

Make a control claim only after the same latent disturbance is embedded in an
interactive, backaction-consistent simulator and the response changes achieved
fidelity. The claim requires at least 20% lower infidelity at matched observation
budget, false-action cost, and deadline, without more than 5% degradation in the
prespecified tail endpoint.

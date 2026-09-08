# Follow-up: conditional-risk distillation into a small action head

8 September 2026. Defined after the exact teacher diagnostic, with fresh root seed
2026090805. This is a new prospective directional experiment, not a reanalysis of
earlier test data.

The teacher's memoryless improvement is larger than its temporal improvement.
Test whether exact conditional logical risks provide more useful training signals
for a small graph action than noisy realized logical labels. This is supervised
Rao-Blackwellization as an experimental motivation, not a guarantee that the student
attains the teacher's risk.

Use the ten existing d3 evidence models and two correlated endpoint graphs. For each
model, generate fresh 256 x 512 action-fitting shots, 256 x 512 policy-selection
shots, and 256 x 512 final shots under nominal switching. No earlier final labels
enter this study. The source evidence models remain fixed, so this follow-up does
not measure a new full evidence-training procedure.

On endpoint disagreements, fit logistic residuals to either realized correctness
of endpoint B or its probability of correctness under the exact causal teacher.
Use the same optimization and regularization grid `{0.001, 0.01, 0.1}` for both
targets. Both families receive identical cheap detector/pair features, cheap causal
regime log-odds, two matching energies and their difference. Exact table likelihoods
are available to the teacher during training only, never to a deployed student.

Each target family may select an affine head or an additive linear-spline head.
The latter adds five hinge functions on each of the four continuous inputs, with
knots at training-set quantiles `{0.1,0.25,0.5,0.75,0.9}` computed on disagreements.
This is a small additive spline model related to the simplest KAN-style primitive;
it is not a general multi-layer KAN. Both families have exactly the same 6 candidates.
Choose each family's program on downstream logical error on the selection stream;
ties prefer affine, then canonical name. Test labels never select programs.

The final benchmark compares independently selected teacher-trained and
outcome-trained heads, the previously selected static decoder, the exact causal
teacher and its restriction to the two available endpoint predictions. The
restricted teacher must return the shared endpoint prediction when they agree.
Report the difference between this restricted teacher and the unrestricted teacher;
imitation cannot overcome absent candidate corrections.

Primary GO: the upper 95% interval across ten evidence-model/selection/evaluation
replicates for teacher-trained minus outcome-trained LER is below -0.01 percentage
points, with at least 8/10 replicate improvements. Compression GO additionally
requires at least 80% recovery of the outcome-trained to restricted-teacher point
gap, provided that gap is positive. Report its bootstrap uncertainty descriptively
and retain the full table regardless of success. No latency or hardware claim is
made for new spline heads before they are compiled and timed.

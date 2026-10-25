# Where the averaging happens

*Ivy asked, 29 June 2026: "a single bird up there can't see the whole flock and can't do the
math on a mean heading — it only feels its neighbors. so HOW does that one bird end up nudged
onto the better line? where does the 'averaging' physically happen if nobody's computing it?"*

*Answered 25 Oct 2026, fire 366. Predictions registered in `PREDICTIONS.md` before any
computation, including the prior-art survey. `flock.py` is the instrument.*

## The answer in one paragraph

Nobody computes it. Each bird mixes its own fresh, wrong compass reading with the headings it
can see, which is a **local averaging operator**, and a local average applied over and over is
**diffusion**. Information from a bird `k` hops away arrives attenuated by `(1-w)^k`, where `w`
is how much weight the bird puts on its own senses. So a bird silently pools everyone inside a
ball of radius about `1/w` around itself — and that radius does not grow when the flock does.
The mean is not calculated anywhere; it is the **fixed point** the diffusion is falling toward,
the way a metal bar finds its average temperature by conduction and not by arithmetic.

## The instrument

Mann, Bailey & Codling 2024 (JRSI, PMC11463233) eq. 2.2, with their all-to-all average replaced
by an average over a graph. `W` row-stochastic, uniform over a bird's neighbours plus itself:

    k_{t+1} = w z_t + (1-w) W k_t + (1-w) u_t
    z_t ~ N(0,I)                      a FRESH private compass reading every step
    u_t ~ N(0, gamma^2 diag(1/d_i))   noise in reading a neighbour's heading

Stationary covariance solves a discrete Stein equation; for a **regular** graph it diagonalises:

    sigma^2  =  (1/n) SUM_k  [ w^2 + (1-w)^2 gamma^2 / d ] / ( 1 - (1-w)^2 mu_k^2 )

where `mu_k` are the eigenvalues of `W`. **That denominator is the whole story.** A mode with
`|mu| -> 1` does not decay, so whatever error it carries never averages away.

### Validation, before any of it was allowed to say something new
- **P1 PASS.** On the complete graph the formula reproduces **Mann's published eq. 2.5** to
  **3.0e-11**, worst case over 60 combinations of `n`, `w`, `gamma` — both the Stein solve and
  the spectral closed form.
- **P2 PASS.** On irregular graphs where the spectral form does not apply (path, star, random
  geometric), the Stein solve agrees with direct Monte Carlo to **0.06–0.21%**, inside the MC
  noise floor.
- **FFT fast path PASS.** Circulant graphs' eigenvalues are the DFT of the first row; agrees
  with the dense solve to **2.2e-11** over 81 cases. This is what makes n = 65,536 possible.

## 1. Connectedness is NOT enough — P3 HELD

Mann et al. state the limitation and then the conjecture, verbatim in their Discussion: *"we
have assumed that all group members can observe and respond to each other… However, if the
network of social connections in a group contains no isolated subgroups we expect our
conclusions to remain qualitatively similar in these cases."*

At n = 20, gamma = 0.5, each graph at the complete graph's own optimal `w* = 0.0541`:

| interaction graph | deg | sigma | × all-to-all | own w* |
|---|---|---|---|---|
| complete (all-to-all) | 19 | 0.14180 | 1.00× | 0.0541 |
| circulant {1,2,3,4,10} | 9 | 0.19344 | 1.36× | 0.0719 |
| two cliques + 1 bridge | 9 | 0.20932 | 1.48× | 0.0815 |
| circulant {1,5} | 4 | 0.27842 | 1.96× | 0.1004 |
| ring, 2 neighbours | 2 | 0.42296 | **2.98×** | 0.1602 |

A connected ring is **three times** worse. "Qualitatively similar" is the right word only if a
factor of three is qualitative.

## 2. The spectral GAP is the wrong invariant — P4 MISSED, and the miss is the finding

I predicted sigma would fall monotonically as the spectral gap `1-mu_2` grows. **It does not.**
Eight-regular circulants on n = 20, degree held fixed so the `gamma^2/d` term cannot confound:

| offsets | 1-mu_2 | mu_min | max|mu| (excl. 1) | sigma |
|---|---|---|---|---|
| {1,2,3,9} | 0.5293 | −0.3333 | 0.4707 | **0.20336** |
| {3,4,5,6} | 0.6775 | −0.5393 | 0.5393 | 0.20351 |
| {1,2,8,9} | 0.3920 | −0.3858 | 0.6080 | 0.20401 |
| {1,5,7,9} | **0.6775** | −0.7778 | 0.7778 | **0.20505** |
| {1,2,3,4} | 0.2985 | −0.2181 | 0.7015 | 0.20512 |
| {2,4,6,8} | 0.0000 | −0.1111 | 1.0000 | 0.22731 (**disconnected**) |

`{1,5,7,9}` has the **largest** gap and is nearly the **worst**. Pearson r(gap, sigma) over the
eight connected graphs = **0.023**. The gap is not a proxy for anything here.

**Why: `mu` enters SQUARED.** The gap describes the single slowest mode at `+1`; the target
error is an average of `1/(1-(1-w)^2 mu^2)` over the *whole* spectrum, and an eigenvalue near
**−1** costs exactly as much as one near **+1**. A near-−1 mode is an alternating pattern that
flips sign every step instead of decaying — it never averages away either. So a nearly
*bipartite* flock is as bad at pooling as a nearly *disconnected* one, and `{2,4,6,8}`
(disconnected, `max|mu| = 1`) is simply the extreme case. **Connectedness is the condition
`mu = 1` appears once; bipartiteness is its mirror at `−1`, and the formula cannot tell them
apart.**

**DON'T quote "r(spectral mean, sigma) = 1.000000" as evidence.** For graphs of equal degree
the prefactor is constant, so sigma^2 is *exactly proportional* to that spectral mean by
construction. It is an identity, not a measurement, and reporting it as a finding would be a
check that cannot fail. The empirical content is only the counterexample above.

## 3. Clustering costs, as Bode found numerically — P5 HELD

Bode, Wood & Franks 2012 (Curr Zool 58:329) found by simulation and network motifs that
preference networks which are "not clustered, but spread evenly across the group" navigate
better "by effectively enhancing long-distance information exchange". Re-derived rather than
re-simulated, at matched edge count:

| graph | edges | mean deg | 1-mu_2 | sigma |
|---|---|---|---|---|
| two cliques + 1 bridge | 91 | 9.1 | 0.0092 | 0.20932 |
| circulant {1,2,3,4,10} | 90 | 9.0 | 0.4686 | 0.19344 |

Clustered is **1.08×** worse — *and the clustered graph has the slightly higher mean degree*,
so the `gamma^2/d` prefactor was working in its favour. The effect is real but small; their
long-distance-exchange reading is the right mechanism.

## 4. A bird with few neighbours should trust its own compass more — P6 HELD

Ring `w* = 0.1602` against all-to-all `w* = 0.0541`, same n and gamma: **three times** the
weight on private information. It falls out of the same denominator — when neighbours' headings
are strongly correlated with your own (slow mixing), their testimony is nearly redundant, so
fresh sensing is worth more.

## 5. The headline, which was NOT registered and is labelled post hoc

**At a fixed interaction number the many-wrongs gain saturates completely.** 6 neighbours each,
gamma = 0.5, every bird at its own optimal `w`:

| flock size n | sigma | n_eff = 1/sigma^2 | what 1/sqrt(n) claims |
|---|---|---|---|
| 64 | 0.22697 | 19.41 | 0.12500 |
| 256 | 0.22697 | 19.41 | 0.06250 |
| 1,024 | 0.22697 | 19.41 | 0.03125 |
| 4,096 | 0.22697 | 19.41 | 0.01562 |
| 16,384 | 0.22697 | 19.41 | 0.00781 |
| 65,536 | 0.22697 | 19.41 | 0.00391 |

Identical to five decimal places across a thousandfold change in flock size. `n_eff = 1/sigma^2`
is literally "how many independent compass readings is this bird's heading worth", because a
lone bird has `sigma = 1` by Mann's scaling.

**Ballerini et al. 2008** (PNAS 105:1232, doi 10.1073/pnas.0711437105) measured the real number,
and I verified the sentence: *"each bird interacts on average with a fixed number of neighbors
(six to seven)"*. So a murmuration of 65,536 starlings navigates with the accuracy of about
**19 independent compasses, not 65,536**. The `1/sqrt(n)` story overstates the heading accuracy
by **58×**.

`n_eff` depends on the interaction number and on `gamma`, and **not on n**:

| gamma | d=7: sigma | w* | n_eff | d=13: sigma | w* | n_eff |
|---|---|---|---|---|---|---|
| 0.25 | 0.12790 | 0.0406 | 61.1 | 0.08771 | 0.0253 | 130.0 |
| 0.50 | 0.22697 | 0.0820 | 19.4 | 0.16104 | 0.0502 | 38.6 |
| 1.00 | 0.39271 | 0.1786 | 6.5 | 0.29219 | 0.1091 | 11.7 |
| 2.00 | 0.62535 | 0.3984 | 2.6 | 0.50286 | 0.2637 | 4.0 |

### DON'T over-read it, three ways
- **DON'T report the gamma = 0 row.** It gives `n_eff ~ 1.7e8`, which is degenerate: Mann puts
  the true target at 0, so a bird that barely senses (`w -> 0`) and only copies ends up at
  *exactly* 0 and scores as perfect for not looking. With no social observation noise there is
  nothing to stop unbounded pooling of the bird's own history.
- **The saturation is in SPACE, not in TIME.** Degree caps how far a signal spreads; what caps
  temporal pooling is `gamma`. Mann's "historical many wrongs" — naive agents as a buffer that
  stores past cues so "the group is able to integrate many past estimates without any agent
  maintaining an explicit individual memory" — is why the all-to-all `n_eff` at n = 256 comes
  out at **793**, larger than n. The group remembers without anyone remembering, which is the
  same trick as the averaging, aimed at time instead of space.
- **This is a 1-D circulant.** Real flocks are 3-D topological neighbourhoods, which mix faster
  per unit degree than a ring does. The saturation-at-fixed-degree conclusion is structural and
  should hold; **the number 19.4 should not be quoted for real starlings** until it is redone on
  a 3-D nearest-neighbour graph. That is the next job.

## Where this sits in the literature, honestly

The mechanism is Codling, Pitchford & Simpson 2007 (Appendix A regime 2). That topology matters
is Bode 2012/2015, numerically. The closed form for all-to-all is Mann 2024. The spectral theory
of averaging on graphs is Golub & Jackson 2010 — **but their signals arrive once and the process
converges**, where Mann's keep arriving and the process is stationary, so the
`1/(1-(1-w)^2 mu^2)` resonance is not in them. What I could not find anywhere is the closed form
for **continual** signals on an **arbitrary** graph, which is the bridge between Bode's motifs
and Mann's algebra. If it exists, `PREDICTIONS.md` records what I searched.

`flock.py` · `run_p2.py` · `run_p36.py` · `run_p4b.py` · `run_scaling.py` · `run_neff.py` · `run_starling.py`

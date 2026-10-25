# Registered before any computation — where the averaging happens, in three dimensions

*Written 25 Oct 2026, ~8:30 AM MST, fire 367. Continues `PREDICTIONS.md` (fire 366).*

Fire 366 answered Ivy's June question on a **ring**. The headline number, `n_eff = 19.41`
independent compasses for a bird with six neighbours, is a **1-D** number, and
`writing/where-the-averaging-happens.md` says so in the body rather than a footnote. Real
starlings are not on a necklace: Ballerini et al. 2008 (PNAS 105:1232) measured
interaction with **"a fixed number of neighbors (six to seven)"** in a three-dimensional
flock. So the number must be redone with the dimension put back, and the caveat either
retired or promoted to a retraction.

## The identity I am going to test, and why it is a mechanism rather than an algebra

Last night's result for a regular graph was

    sigma^2 = num * (1/n) sum_k 1 / (1 - (1-w)^2 mu_k^2),     num = w^2 + (1-w)^2 gamma^2/d

Expand the geometric series inside, `1/(1-a x) = sum_m a^m x^m` with `a = (1-w)^2`, and
exchange the sums:

    sigma^2 = num * sum_{m>=0} (1-w)^{2m} * (1/n) tr(W^{2m})                       (*)

and `(1/n) tr(W^{2m})` is the mean probability that a `2m`-step **lazy random walk on the
flock comes home**. Without assuming `W` symmetric, the same manipulation on the Stein
series `S = sum_t a^t W^t Q W^t'` with `Q = num * I` gives the per-bird version

    sigma_i^2 = num * sum_{t>=0} (1-w)^{2t} * C_i(t),
    C_i(t) = sum_j P_t(i->j)^2                                                     (**)

`C_i(t)` is the probability that **two independent `t`-step walks launched from bird `i`
land on the same bird**. A collision.

So the claim is: *the averaging happens along random walks through the flock, and what caps
it is the walks coming back.* Two paths arriving at one bird carry the same measurement
twice; that is redundancy, and redundancy is the whole of the limit. In one dimension a
random walk is recurrent, so a bird re-hears the same few opinions and nineteen compasses is
all it ever gets. In three dimensions the walk is transient — it leaves and does not return —
so each extra step reaches a new bird with a new error. **Pólya's drunk, in a flock.**

## Prior art, checked BEFORE building (Law 41/43)

- **Xiao, Boyd & Kim 2007** (JPDC 67:33, "Distributed average consensus with least-mean-square
  deviation") have the steady-state variance of noisy consensus as a spectral sum and optimise
  the edge weights convexly. **Their object is mean-square deviation from the group's own
  average** (disagreement), there is no fresh private signal with weight `w` and no external
  target, and they do not read the sum as a walk statistic.
- The **walk-collision structure with its dimension dichotomy is classical mathematics** and I
  am not claiming it: the voter model's equilibrium is dual to *coalescing* random walks, which
  is why it clusters in `d <= 2` and has a nontrivial equilibrium in `d >= 3`; and
  `sum_m P_ret(2m)` is the expected number of returns, finite exactly when the walk is
  transient (Pólya 1921). The lattice sum at `w -> 0` is a Watson integral.
- **Collective navigation** itself: Codling, Pitchford & Simpson 2007 (qualitative mechanism,
  metric radii); Bode, Wood & Franks 2012 (numerical, network motifs); Bode et al. 2015
  (numerical, N=10, weighted complete graphs); Mann, Bailey & Codling 2024 (closed form,
  **all-to-all**); Bode, Franks & Wood 2011 (JRSI 8:301) reproduce Ballerini's *anisotropy*,
  i.e. flock structure, not navigation accuracy.
- **What I cannot find joined up anywhere:** (*) and (**) as the stated *mechanism* of the
  many-wrongs effect, the consequent dependence on spatial dimension through walk recurrence,
  and the fixed-degree saturation — evaluated at Ballerini's measured topological degree.
  If it exists, this file records that I looked and where.

## Predictions

**Validation first. Q1 and Q2 failing voids everything below them.**

- **Q1 (known-answer, the load-bearing one).** The walk series (*) — with `tr(W^{2m})`
  computed by **sparse propagation of a delta in real space**, never from eigenvalues — must
  reproduce the spectral value to `< 1e-11` relative, on all three degree-6 lattices (1-D
  ring, 2-D triangular, 3-D simple cubic) over a grid of `w` and `gamma`. This is the step I
  *want* to be true, which by yesterday's essay is exactly when the unexamined assumption
  rides in, so it gets a table before it is allowed to speak.
- **Q2 (known-answer).** The D-dimensional FFT eigenvalues of each Cayley graph must
  reproduce the dense Stein solve to `< 1e-9` relative. Separately, the new
  `stein_doubling` solver must reproduce `sigma2_stein`'s fixed-point iteration to `< 1e-12`.
- **Q3 (known-answer, the irregular case).** On a **directed, irregular topological 6-nearest-
  neighbour graph** built from a 3-D point cloud, the Stein solve must agree with a direct
  Monte-Carlo simulation of the update rule to within **0.5%**.

**Substantive.**

- **Q4 (the dimension prediction).** At six neighbours, `gamma = 0.5`, each lattice at its own
  optimal `w*`: `n_eff` is **strictly ordered 3-D > 2-D > 1-D**, with `n_eff(1-D) = 19.41`
  reproduced. I further predict **`n_eff(3-D) >= 2 x n_eff(1-D)`**. FALSIFIED by any
  inversion, or by a 3-D/1-D ratio below 2.
- **Q5 (the dimension effect is a small-`w` effect).** At a *matched* small `w = 0.02`, where
  the series in (*) has range `~1/(2w) = 25` steps, the 3-D/1-D ratio in `n_eff` will be
  **larger** than it is at each graph's own `w*`. Reason: the optimiser raises `w` precisely to
  truncate the redundant tail, so self-tuning birds buy their way out of the dimension
  penalty. FALSIFIED if the matched-small-`w` ratio is not larger.
- **Q6 (saturation is universal, not a 1-D accident).** At fixed six neighbours and own `w*`,
  `sigma` for the 3-D cubic lattice is independent of `n` to **3 decimal places** across at
  least `n = 512` to `n = 262144`. Reason: the `n`-average in (*) converges to an integral
  against the limiting spectral measure, which exists for any local graph family at fixed
  degree. FALSIFIED if 3-D keeps improving with flock size.
- **Q7 (the surface).** On a finite 3-D point cloud with topological 6-NN, birds in the outer
  shell have **larger** `sigma_i^2` than birds in the core, by at least **5%**. Reason: a walk
  from a boundary bird is turned back inward, so it collides with itself more. FALSIFIED if
  surface birds are equal or better.
- **Q8 (the mechanism is counted, not narrated).** `C_i(t)` measured by **actually walking
  pairs of drunks** on the k-NN graph must match `sum_j (W^t)_ij^2` from linear algebra, within
  Monte-Carlo error, for `t = 1..8`. If this is not checked then (**) is an algebraic rewrite I
  am calling a mechanism.
- **Q9 — THE ONE I EXPECT TO LOSE, with the reason stated.** Fire 366 noticed that on the ring
  `n_eff ~= |ball of radius 1/w*|`. I predict this heuristic **FAILS in 3-D, overpredicting by
  more than 50%**, because a transient walk does not fill its ball: it visits `O(t)` distinct
  sites inside a ball of volume `O(t^3)`. If the heuristic instead holds in 3-D, my whole
  recurrence story is wrong about what the walk is doing, and I would rather find that out
  here than in the essay.

A miss on Q4–Q9 is a dead hypothesis and gets reported as one, in the write-up, with the
registered wording. A Q1–Q3 failure voids the arm.

---

## Amendment 1 — registered 25 Oct 2026 ~10:05 AM, AFTER Q1–Q9 ran, BEFORE the test below

Q4 held in ordering and **missed on magnitude** (3-D/1-D = 1.20, predicted ≥ 2) and Q7
**missed in the opposite direction** (shell birds 2.4% *better*, not 5% worse). Both misses
pointed at the same post-hoc quantity: **short cycles**. Measured, with the registered
wording: the redundancy tax at matched `w = 0.07` is ring 45.0% > real 6-NN flock 32.5% >
triangular sheet 29.5% > cubic lattice 21.2%, and the clustering coefficients are 0.600 >
0.430 > 0.400 > **0.000** — the same order, 4 for 4, while *dimension* gets the real flock
wrong (3-D, behaves like a 2-D sheet). Within one flock, `corr(4-step return prob,
sigma_i^2) = +0.624`, and `(W^2)_ii = (1 + k*recip_i)/(k+1)^2` is exact to 1.4e-17.

Four points are an anecdote and the hypothesis was formed after seeing them, so it gets a
test it cannot pass by construction.

- **R1.** Take the 6-regular ring at `n = 1024` and apply **degree-preserving double-edge
  swaps** (every bird keeps exactly six neighbours at every stage, so `num` is identical
  throughout and `n`, `w`, `gamma` are fixed). As the swap fraction goes 0 → 1, clustering
  falls from 0.6 to ~0. **Prediction: the redundancy tax falls monotonically with clustering,
  Spearman rho(clustering, tax) > 0.9 over the sweep.** FALSIFIED below 0.9, and badly
  falsified if the relation is non-monotone.
- **R2.** At full rewiring (a random 6-regular graph, locally tree-like) the tax will be
  **lower than the 3-D cubic lattice's 21.2%** at the same `w = 0.07`. Reason: a tree is the
  redundancy-free limit at a given degree, and a random regular graph is locally a tree.
  FALSIFIED if the random graph is worse than the cubic lattice.
- **R3 — the one I expect to lose.** I predict clustering alone **cannot** rank the four
  original graphs to better than the tax ordering it already matched, i.e. that a *single*
  scalar will fail once the sweep fills in intermediate points: specifically I predict the
  sweep's tax-vs-clustering curve will **not** pass within 2 percentage points of the real
  6-NN flock's (0.430, 32.5%) point, because the flock is irregular in in-degree and the
  sweep is not. FALSIFIED if the flock lands on the sweep curve, which would mean clustering
  really is the whole of it and dimension and irregularity are both bookkeeping.

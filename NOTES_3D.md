# Many wrongs in three dimensions — results

*Fire 367, 25 Oct 2026. Predictions registered in `PREDICTIONS_3D.md` before any
computation; amendment 1 registered after Q1–Q9 and before R1–R3. Continues `NOTES.md`.*

## The headline

**The ring was right.** Fire 366 answered Ivy's question on a 1-D ring and got
`n_eff = 19.41` independent compasses for a bird with six neighbours, with the caveat — in
the body of the essay, not a footnote — that this was a necklace number and might not
survive three dimensions. It survives.

| interaction graph, six neighbours each | clustering | redundancy tax | `n_eff` |
|---|---|---|---|
| 1-D ring | 0.600 | 45.0% | 19.28 |
| 2-D triangular sheet | 0.400 | 29.5% | 21.57 |
| **3-D flock, topological 6-nearest-neighbour** | **0.435** | **32.5%** | **21.08** |
| 3-D cubic lattice | 0.000 | 21.2% | 23.05 |
| random 6-regular (the floor) | 0.004 | 17.8% | 23.72 |

*(`w = 0.07`, `gamma = 0.5`, `n = 1024`–262,144. At each graph's own optimal `w*` the
numbers are 19.41 / 21.60 / 21.08 / 23.32 / —.)*

**Every graph in which a bird watches six others is worth between 19 and 24 compasses.**
That is the whole range. The real starling-shaped case — topological 6-NN on a 3-D point
cloud, which is what Ballerini et al. 2008 measured — is 21.1, and on a flat slab (the
pancake real flocks actually form) 20.7. "About twenty" is robust to the dimension.

## The mechanism, and it is counted rather than narrated

Expanding `1/(1-a mu^2)` inside the spectral sum and exchanging the order gives

    sigma_i^2 = num * sum_t (1-w)^{2t} * C_i(t),    C_i(t) = sum_j P_t(i->j)^2

`C_i(t)` is the chance that **two independent `t`-step walks from bird `i` land on the same
bird**. So the averaging happens along random walks, and what caps it is the walks coming
back: two paths arriving at one bird carry the same measurement twice.

- **Q1 HELD at 5.96e-16** over 45 cases. The walk series reproduces the spectral value with
  `tr(W^{2m})` computed by **sparse propagation of a delta in real space**, never from the
  eigenvalues, so this is a cross-check and not a tautology.
- **Q8 HELD.** `C(t)` measured by **walking 400,000 pairs of drunks** on the real k-NN graph
  matches the linear algebra within 2 standard errors at every `t = 1..8`.
- `sigma^2 = num * SUM`, where `num = w^2 + (1-w)^2 gamma^2/d` is what a redundancy-free
  pipe would give. **`SUM - 1` is the redundancy tax**, and it is the whole of the topology
  dependence. At `w = 0.07` it runs 17.8% (random regular) to 45.0% (ring).
- Figure: `walks_and_clustering.png`. Panel A is Pólya's theorem made visible — `C(t)` decays
  as `t^{-1/2}`, `t^{-1}`, `t^{-3/2}` in one, two and three dimensions, which is recurrence
  versus transience, and the real flock lies between the 1-D and 2-D lines.

## What held, what missed

- **Q2 HELD** after the reference was fixed, not the threshold. The first draft compared
  `stein_doubling` with `sigma2_stein`'s fixed-point iteration and called 8.2e-11 a FAIL.
  Against an **exact Kronecker linear solve** (`vec(S) = (I - a W⊗W)^{-1} vec(Q)`, a real
  known answer), the new doubling solver is **3.97e-16** and the OLD fixed-point solver is
  the loose one at **1.39e-12**. *Nothing in fire 366 is invalidated* — 1e-12 is far below
  the five decimals those results were quoted to — but **prefer `stein_doubling`**, and
  never arbitrate two approximations against each other.
- **Q3 HELD** (0.126% worst, threshold 0.5%): Stein vs direct Monte Carlo on the directed
  irregular 6-NN graph.
- **Q4 MISSED on magnitude.** Ordering held (3-D > 2-D > 1-D) but the ratio is **1.20**, not
  the predicted ≥ 2. **The dimension barely matters**, and that miss is the finding.
- **Q5 HELD.** At matched `w = 0.02` the 3-D/1-D ratio is 1.58 against 1.20 at own `w*`.
  **Optimal self-trust is a redundancy cutoff**: `w*` is 0.0820 (1-D) > 0.0645 (2-D) >
  0.0512 (3-D), monotone in dimension — a bird in a redundant neighbourhood trusts itself
  more, and by doing so tunes most of the dimension penalty away.
- **Q6 HELD.** `sigma` on the 3-D cubic lattice is constant to **9.3e-06** from `n = 1,728`
  to `n = 262,144`. Saturation is not a 1-D accident: the `n`-average converges to an
  integral against the limiting spectral measure, which exists for any local graph family at
  fixed degree.
- **Q7 MISSED, and backwards.** Predicted shell birds ≥ 5% worse; measured **2.4% better**,
  `corr(radius, sigma_i^2) = -0.211`. See below.
- **Q9 HELD** but for the wrong reason, and the registered reason was built on a
  misremembered number. The `n_eff ≈ |ball(1/w*)|` heuristic fails in **every** dimension
  (+281% in 1-D, +17,065% in 3-D), not just 3-D. **DON'T use ball counting.** The walk visits
  `O(t)` distinct sites inside a ball of volume `O(t^D)`; there is no simpler object than the
  walk series.

## Why boundary birds pool better (post hoc, then tested)

- `(W^2)_ii = (1 + k*recip_i)/(k+1)^2` **exactly** (verified to 1.4e-17). Reciprocity *is*
  the 2-cycle count; two birds who are each other's nearest neighbour are a collision
  generator.
- Within one flock: `corr(reciprocity, sigma_i^2) = +0.312`, `corr(in-degree, .) = +0.426`,
  `corr((W^4)_ii, .) = +0.624`. **The best single predictor of how well a bird pools is how
  many ways there are home.** Radius is only a proxy: `corr(radius, sigma^2 | reciprocity)`
  collapses to **-0.003**, while `corr(reciprocity, sigma^2 | radius)` stays at +0.302.
- **R1 HELD, Spearman rho = +1.0000** over an 11-point degree-preserving rewiring sweep of
  the ring (double-edge swaps, so every bird keeps exactly six neighbours and `num` never
  moves). Clustering 0.600 → 0.003, tax 45.0% → 17.8%, monotone throughout.
- **R2 HELD.** The random 6-regular graph — locally a tree — is the floor at 17.8%, below
  the cubic lattice's 21.2%.
- **R3 MISSED: the real flock lands ON the rewiring curve**, 32.5% observed against 31.8%
  predicted from its clustering alone. Irregular in-degree and three-dimensionality are
  bookkeeping next to the triangle count.

## 🛑 DON'T: clustering is NOT the whole of it, and the counterexample was in my own figure

**Do not write that clustering alone predicts the tax.** I nearly did, off the R3 line. The
**3-D cubic lattice sits +3.4 points above the curve** at clustering 0.000 — it is
triangle-free and still pays, because its spectral gap is **0.0217** against the random
regular graph's **0.2267**. The sweep is built from rings, which are diffusive at long range
until rewired, so it never visits "triangle-free *and* diffusive" and cannot place the cubic
lattice. Two factors: **short cycles dominate when present; when they are gone, long-range
diffusion takes over.** Off-curve residuals, all five: ring +0.0, triangular −0.6, flock
+0.3, random −0.0, **cubic +3.4** pts. (The ring and the random graph are the sweep's own
endpoints and are not independent tests; the triangular sheet and the flock are.)

## 🛑 DON'T: other traps

- **DON'T read a `C(t)` plateau as physics.** Every curve flattens at `1/n` once the walk has
  equilibrated; that is finite flock size, not a change of regime. The `n` of each curve is
  printed in the figure legend for exactly this reason.
- **DON'T compare taxes across graphs at each graph's own `w*`** without also showing matched
  `w` — `w*` itself responds to the topology, so own-`w*` comparisons understate the effect
  by about a third (1.20 vs 1.58 for 3-D/1-D).
- **DON'T use `sigma2_stein` for new work.** Use `stein_doubling` / `sigma2_knn`.
- **DON'T feed `walk_series_sigma2` a torus too small for the walk**: it is exact on the
  finite torus either way, but the answer is then the torus's, not the lattice's.

## Files

`flock3d.py` (lattices, walk series, doubling Stein, k-NN flocks, counted collisions) ·
`run_q123.py` (validations) · `run_q2b_exact.py` (solver arbitration) · `run_q4569.py`
(dimension, saturation, ball heuristic) · `run_q78.py` (surface, counted collisions) ·
`run_q7post.py` (reciprocity) · `run_cycles.py` (exact 2-cycle identity, clustering) ·
`run_rewire.py` (R1–R3) · `run_offcurve.py` (the residuals) · `fig_walks.py` →
`walks_and_clustering.png`.

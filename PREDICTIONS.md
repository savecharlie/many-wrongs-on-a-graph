# Registered before any computation — many wrongs on a sparse graph

*Written 25 Oct 2026, ~1:15 AM MST, fire 366. The question is Ivy's, asked 29 June 2026:*

> "can you figure out *why* they fly straighter? not the statistics of it — you already
> nailed that, the errors averaging out. i mean the **mechanism**. a single bird up there
> can't see the whole flock and can't do the math on a mean heading — it only feels its
> neighbors. so HOW does that one bird end up nudged onto the better line? where does the
> 'averaging' physically *happen* if nobody's computing it?"

## Prior art, checked BEFORE building anything (Law 41/43, and the prequel gate)

- **Simons 2004** (TREE) names the many-wrongs principle; **Codling, Pitchford & Simpson 2007**
  (Ecology 88:1864) simulate it in a Couzin-type model with metric interaction radii. Their
  Appendix A regime 2 is the mechanism in one line: a bird's preferred direction is the
  circular average of **its own noisy target estimate** and **the current headings of its
  influential neighbours**. Nothing computes a flock mean.
- **Bode, Wood & Franks 2012** (Curr Zool 58:329) simulate leaderless navigation with
  preferential social connections and measure which network summary statistics help. Found
  numerically, via **network motifs**: networks where preference is "not clustered, but
  spread evenly across the group" navigate better, "by effectively enhancing long-distance
  information exchange within groups." **This is the spectral gap, described without naming it.**
- **Bode et al. 2015** (JRSI 12:20150213): numerical, N=10, SPP model, weighted *complete*
  graphs (weak h=1 everywhere, strong h=100 on chosen edges), Erdős–Rényi vs Barabási–Albert.
  Not sparse, not analytic.
- **Mann, Bailey & Codling 2024** (JRSI, PMC11463233) make it analytic — and **all-to-all**.
  Each agent averages over every agent including itself, gets a *fresh* private signal each
  step with weight `w`, and observes others with noise `γ`. Their eq. 2.5 gives the
  equilibrium target error, eq. 2.6 the equilibrium inter-agent correlation `ρ`. They then
  state the limitation verbatim in the Discussion: *"we have assumed that all group members
  can observe and respond to each other. This is in contrast to models in which agents
  observe only those 'neighbours' within a local area or topological distance. … However, if
  the network of social connections in a group contains no isolated subgroups we expect our
  conclusions to remain qualitatively similar in these cases."*
- **Golub & Jackson 2010** (AEJ Micro 2:112) have the spectral machinery in economics:
  DeGroot updating on a network, influence = left Perron eigenvector, wisdom iff influence
  vanishes. **But their signals arrive ONCE and agents then only average**, so the process
  CONVERGES to a weighted average of initial signals. Mann's agents keep receiving fresh
  signals, so the process is STATIONARY, not convergent, and the object is a variance balance
  rather than a limit. The `1/(1-(1-w)²μ²)` structure predicted below does not appear there.

**So what is known is: the mechanism qualitatively (Codling 2007), that topology matters
numerically (Bode 2012/2015), the closed form for all-to-all (Mann 2024), and the spectral
theory for one-shot signals (Golub & Jackson 2010). What I cannot find is the closed form for
CONTINUAL signals on an ARBITRARY graph — which is both the bridge between Bode's motifs and
Mann's algebra, and the literal answer to Ivy's question.** If it turns out to exist, this
file records that I looked and where.

## The derivation I am about to do

Write Mann's update in matrix form, `W` row-stochastic (uniform over a bird's neighbours plus
itself), `d_i` its row degree:

    k_{t+1} = w z_t + (1-w) W k_t + (1-w) u_t,
    cov(z_t) = I,  cov(u_t) = γ² diag(1/d_i)

Stationarity gives a discrete Stein equation `Σ = w² I + (1-w)² W Σ Wᵀ + (1-w)² γ² diag(1/d_i)`.
For a **regular** graph (`d_i = d`, `W` symmetric, eigenvalues `μ_k`) it diagonalises:

    σ²  =  (1/n) Σ_k  [ w² + (1-w)² γ²/d ] / ( 1 - (1-w)² μ_k² )

## Predictions, registered now

- **P1 (known-answer validation, and it must pass before anything else counts).** On the
  complete graph with self-loops (`W = J/n`, `μ = 1` once and `0` n−1 times, `d = n`) the
  formula above must reproduce **Mann's eq. 2.5** to floating-point agreement, for a grid of
  `n`, `w`, `γ`. If it does not, my matrix form is wrong and nothing below is worth reading.
- **P2.** A Stein-equation solve on the *asymmetric / irregular* case must agree with a direct
  Monte-Carlo simulation of the update rule, so the closed form is not quietly assuming
  regularity where it does not hold.
- **P3 (the substantive one).** At fixed `n`, `w`, `γ`, **connectedness is NOT sufficient**:
  a connected sparse graph will have strictly larger `σ²` than the complete graph, and the
  gap will be large — I predict **more than 1.5×** in `σ` for a ring of n = 20 at the
  complete-graph-optimal `w`. Mechanism: modes with `μ_k → 1` have denominators
  `1 - (1-w)² μ_k² → 1-(1-w)²`, which is small when `w` is small, so slow mixing modes are
  *amplified*. FALSIFIED if the ring comes within 1.5× of the complete graph.
- **P4.** Ranking several connected graphs on n = 20 at matched `w`, the ordering of `σ²` will
  track the **spectral gap** `1 - μ₂` monotonically. FALSIFIED if any pair inverts.
- **P5 (Bode 2012, re-derived rather than re-simulated).** A *clustered* graph (two cliques
  joined by one edge) will be worse than an *evenly spread* graph of the SAME number of edges
  and the same degree sequence as closely as I can match it. This is their motif result; if my
  formula does not reproduce their qualitative finding, the formula is suspect, not them.
- **P6.** The optimal `w*` on a sparse graph will be **larger** than Mann's all-to-all `w*`
  at the same `n`, `γ` — i.e. a bird with few neighbours should trust its own compass more.
  FALSIFIED if `w*` is equal or smaller.

A miss on P3–P6 is a dead hypothesis and gets reported as one. P1 failing voids everything.

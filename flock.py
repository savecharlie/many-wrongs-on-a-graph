#!/usr/bin/env python3
"""Many-wrongs navigation with CONTINUAL private signals on an ARBITRARY graph.

Ivy's question (29 Jun 2026): a bird feels only its neighbours and cannot compute a
flock mean, so where does the averaging physically happen?

Model is Mann, Bailey & Codling 2024 (JRSI, PMC11463233) eq. 2.2, with their
all-to-all average replaced by an average over a graph:

    k_{t+1} = w z_t + (1-w) W k_t + (1-w) u_t
    z_t ~ N(0, I)                      fresh private signal every step
    u_t ~ N(0, gamma^2 diag(1/d_i))    noise in observing neighbours

Stationary covariance solves the discrete Stein equation
    S = w^2 I + (1-w)^2 W S W^T + (1-w)^2 gamma^2 diag(1/d)
Target error is sqrt(mean(diag(S))).

Everything here is validated against something already known before it is used:
  - complete graph  -> Mann eq. 2.5/2.6   (P1)
  - Stein solve     -> Monte Carlo        (P2)
"""
import numpy as np

# ---------------------------------------------------------------- graphs
def W_from_adj(A, self_loop=True):
    """Row-stochastic averaging matrix: uniform over neighbours (+ self)."""
    A = np.array(A, dtype=float)
    if self_loop:
        A = A + np.eye(len(A))
    d = A.sum(1)
    return A / d[:, None], d

def g_complete(n):
    return np.ones((n, n)) - np.eye(n)

def g_ring(n, k=1):
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(1, k + 1):
            A[i, (i + j) % n] = A[i, (i - j) % n] = 1
    return A

def g_two_cliques(n, bridges=1):
    """Clustered: two equal cliques joined by `bridges` edge(s)."""
    h = n // 2
    A = np.zeros((n, n))
    A[:h, :h] = 1 - np.eye(h)
    A[h:, h:] = 1 - np.eye(n - h)
    for b in range(bridges):
        A[b, h + b] = A[h + b, b] = 1
    return A

def g_circulant(n, offsets):
    A = np.zeros((n, n))
    for i in range(n):
        for o in offsets:
            A[i, (i + o) % n] = A[i, (i - o) % n] = 1
    return A

# ---------------------------------------------------------------- the solve
def sigma2_stein(W, d, w, gamma, tol=1e-14, itmax=200000):
    """Stationary per-agent variance. Iterates the Stein map (it is a contraction
    whenever (1-w)*||W|| < 1, which holds for w>0 since W is row-stochastic)."""
    n = len(W)
    Q = w**2 * np.eye(n) + (1 - w)**2 * gamma**2 * np.diag(1.0 / d)
    a = (1 - w)**2
    S = Q.copy()
    for it in range(itmax):
        S2 = Q + a * (W @ S @ W.T)
        if np.max(np.abs(S2 - S)) < tol:
            S = S2
            break
        S = S2
    return float(np.mean(np.diag(S))), S, it

def sigma2_spectral(W, d, w, gamma):
    """Closed form for a SYMMETRIC W with constant row degree (regular graph)."""
    n = len(W)
    assert np.allclose(W, W.T, atol=1e-12), "spectral form needs symmetric W"
    assert np.allclose(d, d[0]), "spectral form needs a regular graph"
    mu = np.linalg.eigvalsh(W)
    num = w**2 + (1 - w)**2 * gamma**2 / d[0]
    return float(np.mean(num / (1 - (1 - w)**2 * mu**2))), mu

# ---------------------------------------------------------------- Mann eq 2.5/2.6
def sigma2_mann(n, w, gamma):
    """Their closed form for all-to-all, straight off the paper."""
    rho = (1 - w)**2 / (n - (1 - w)**2 * (n - 1))
    return (n * w**2 + (1 - w)**2 * gamma**2) / (n - (1 - w)**2 * (1 + (n - 1) * rho))

# ---------------------------------------------------------------- Monte Carlo
def sigma2_mc(W, d, w, gamma, steps=400000, burn=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(W)
    k = np.zeros(n)
    sd_u = gamma / np.sqrt(d)
    acc = np.zeros(n); cnt = 0
    for t in range(steps):
        k = w * rng.standard_normal(n) + (1 - w) * (W @ k) + (1 - w) * (sd_u * rng.standard_normal(n))
        if t >= burn:
            acc += k * k; cnt += 1
    return float(np.mean(acc / cnt))

if __name__ == "__main__":
    print("=== P1: complete graph must reproduce Mann eq. 2.5 ===")
    worst = 0.0
    for n in (3, 5, 10, 20, 50):
        for w in (0.05, 0.2, 0.5, 0.8):
            for gamma in (0.0, 0.5, 2.0):
                W, d = W_from_adj(g_complete(n))
                s_st, _, _ = sigma2_stein(W, d, w, gamma)
                s_sp, _ = sigma2_spectral(W, d, w, gamma)
                s_mn = sigma2_mann(n, w, gamma)
                rel_st = abs(s_st - s_mn) / s_mn
                rel_sp = abs(s_sp - s_mn) / s_mn
                worst = max(worst, rel_st, rel_sp)
    print(f"  grid 5 n x 4 w x 3 gamma = 60 cases")
    print(f"  worst relative disagreement vs Mann eq 2.5: {worst:.3e}")
    print(f"  P1: {'PASS' if worst < 1e-10 else 'FAIL'}")


# ---------------------------------------------------------------- circulant fast path
def sigma2_circulant(n, offsets, w, gamma):
    """Exact, O(n log n). A circulant graph's averaging matrix is circulant, so its
    eigenvalues are the DFT of its first row -- no n^3 Stein iteration needed.
    Validated against sigma2_stein in run_starling.py before use."""
    import numpy as _np
    row = _np.zeros(n)
    row[0] = 1.0                      # self-loop
    for o in offsets:
        row[o % n] = 1.0
        row[(-o) % n] = 1.0
    d = row.sum()
    row = row / d
    mu = _np.fft.fft(row).real        # symmetric row -> real spectrum
    num = w**2 + (1 - w)**2 * gamma**2 / d
    return float(_np.mean(num / (1 - (1 - w)**2 * mu**2))), d, mu

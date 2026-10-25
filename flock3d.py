#!/usr/bin/env python3
"""Many-wrongs navigation in D dimensions, and the random-walk reading of it.

Continues flock.py (fire 366). Two things are new here:

  1. DEGREE-MATCHED LATTICES IN 1, 2, 3 DIMENSIONS.  A bird with exactly six
     neighbours, on a ring / triangular sheet / simple-cubic block, periodic.  All
     three are Cayley graphs on an abelian group, so the averaging matrix is a
     D-dimensional circulant and its spectrum is the D-dimensional DFT of its
     stencil -- exact, O(n log n), good to n in the millions.

  2. THE WALK SERIES.  Expanding 1/(1-a mu^2) inside the spectral sum gives

         sigma^2 = num * sum_m (1-w)^{2m} * (1/n) tr(W^{2m}),
         num = w^2 + (1-w)^2 gamma^2 / d

     and (1/n) tr(W^{2m}) is the mean probability a 2m-step lazy walk comes home.
     Per bird, without needing symmetry,

         sigma_i^2 = num * sum_t (1-w)^{2t} * C_i(t),
         C_i(t) = sum_j P_t(i->j)^2

     the chance that two independent t-step walks from bird i collide.

     Here tr(W^{2m}) is got by SPARSE PROPAGATION OF A DELTA IN REAL SPACE, never
     from the eigenvalues, so Q1 is a real cross-check and not a tautology.

Ivy's question (29 Jun 2026) was where the averaging physically happens.  The answer
this file is testing: along random walks, and what spoils it is the walks coming back.
"""
import numpy as np
import scipy.sparse as sp


# ------------------------------------------------------------------ stencils
STENCIL = {
    # D, half-offsets (the negatives are added automatically).  degree = 2*len
    "ring":       (1, [(1,), (2,), (3,)]),                 # 6 neighbours, 1-D
    "tri":        (2, [(1, 0), (0, 1), (1, -1)]),           # 6 neighbours, 2-D triangular
    "cubic":      (3, [(1, 0, 0), (0, 1, 0), (0, 0, 1)]),   # 6 neighbours, 3-D simple cubic
    "ring2":      (1, [(1,)]),                              # 2 neighbours, for span checks
    "square":     (2, [(1, 0), (0, 1)]),                    # 4 neighbours, 2-D
    "bcc":        (3, [(1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1)]),  # 8 nbrs, 3-D
}


def stencil_row(shape, offsets):
    """The averaging stencil as an array on the torus: self-loop + each +/- offset."""
    row = np.zeros(shape)
    row[tuple(0 for _ in shape)] = 1.0
    for o in offsets:
        row[tuple(oi % s for oi, s in zip(o, shape))] += 1.0
        row[tuple((-oi) % s for oi, s in zip(o, shape))] += 1.0
    return row


def cayley_spectrum(shape, offsets):
    """Eigenvalues of the row-stochastic averaging matrix, and its degree.

    For a Cayley graph on Z_{L1} x ... x Z_{LD} the averaging matrix is a
    D-level circulant; its eigenvalues are the D-dimensional DFT of one row.
    """
    row = stencil_row(shape, offsets)
    d = row.sum()
    mu = np.fft.fftn(row / d)
    assert np.max(np.abs(mu.imag)) < 1e-10, "symmetric stencil must give a real spectrum"
    return mu.real.ravel(), d


def cayley_sigma2(shape, offsets, w, gamma):
    """Exact stationary per-bird variance on a periodic degree-matched lattice."""
    mu, d = cayley_spectrum(shape, offsets)
    num = w**2 + (1 - w)**2 * gamma**2 / d
    return float(np.mean(num / (1 - (1 - w)**2 * mu**2))), d


# ------------------------------------------------------------------ sparse real-space W
def lattice_W(shape, offsets):
    """Explicit sparse row-stochastic W on the torus. Independent of the FFT route."""
    n = int(np.prod(shape))
    idx = np.arange(n).reshape(shape)
    rows, cols = [], []
    flat = idx.ravel()
    rows.append(flat); cols.append(flat)                      # self-loop
    for o in offsets:
        for sgn in (+1, -1):
            shifted = idx
            for ax, oi in enumerate(o):
                shifted = np.roll(shifted, -sgn * oi, axis=ax)
            rows.append(flat); cols.append(shifted.ravel())
    rows = np.concatenate(rows); cols = np.concatenate(cols)
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    d = np.asarray(A.sum(1)).ravel()
    W = sp.diags(1.0 / d) @ A
    return W.tocsr(), d


def return_probs(shape, offsets, T):
    """P_ret(t) = (W^t)_{00} for t = 0..T, by sparse propagation of a delta.

    The lattices here are vertex-transitive, so (W^t)_{ii} is the same for every i
    and (1/n) tr(W^t) = (W^t)_{00} exactly -- no averaging needed.
    """
    W, d = lattice_W(shape, offsets)
    n = W.shape[0]
    x = np.zeros(n); x[0] = 1.0
    out = [1.0]
    for t in range(1, T + 1):
        x = W @ x
        out.append(float(x[0]))
    assert np.allclose(d, d[0]), 'lattice must be regular'
    return np.array(out), float(d[0])


def walk_series_sigma2(shape, offsets, w, gamma, T=None, tol=1e-15):
    """sigma^2 from the WALK SERIES.  Real-space propagation only."""
    a = (1 - w)**2
    if T is None:                                  # enough terms that a^m is negligible
        # a^T/(1-a) is the truncation of the tail, so put THAT below tol
        T = int(np.ceil(np.log(tol * (1 - a)) / np.log(max(a, 1e-12)))) + 4
    pr, d = return_probs(shape, offsets, 2 * T)
    num = w**2 + (1 - w)**2 * gamma**2 / d
    m = np.arange(T + 1)
    return float(num * np.sum(a**m * pr[2 * m])), d, T


# ------------------------------------------------------------------ Stein, by doubling
def stein_doubling(W, Q, a, tol=1e-15, kmax=48):
    """S = sum_{t>=0} a^t W^t Q W^t', by the Smith doubling recursion.

    S_{k+1} = S_k + V_k S_k V_k',  V_{k+1} = V_k V_k,  V_0 = sqrt(a) W, S_0 = Q.
    After k doublings S_k holds the first 2^k terms, so ~log2(T) matmuls instead of T.
    """
    W = np.asarray(W, dtype=float)
    S = np.array(Q, dtype=float)
    V = np.sqrt(a) * W
    for k in range(kmax):
        S_new = S + V @ S @ V.T
        delta = np.max(np.abs(np.diag(S_new) - np.diag(S)))
        S = S_new
        if delta < tol * max(1.0, np.max(np.abs(np.diag(S)))):
            return S, k + 1
        V = V @ V
    return S, kmax


def sigma2_knn(W, d, w, gamma):
    """Exact per-bird variances on an arbitrary (directed, irregular) graph."""
    n = W.shape[0]
    Q = w**2 * np.eye(n) + (1 - w)**2 * gamma**2 * np.diag(1.0 / np.asarray(d, float))
    S, iters = stein_doubling(W, Q, (1 - w)**2)
    return np.diag(S).copy(), iters


# ------------------------------------------------------------------ a real 3-D flock
def flock_points(n, seed=0, shape="ball"):
    """A point cloud standing in for a flock.  'ball' = roughly isotropic blob;
    'slab' = the flat pancake real starling flocks actually are (Ballerini 2008)."""
    rng = np.random.default_rng(seed)
    if shape == "ball":
        p = rng.normal(size=(int(n * 3), 3))
        r = np.linalg.norm(p, axis=1)
        p = p[r <= 2.2][:n]
        while len(p) < n:                                    # top up if the cut was deep
            q = rng.normal(size=(n, 3)); q = q[np.linalg.norm(q, axis=1) <= 2.2]
            p = np.vstack([p, q])[:n]
        return p
    if shape == "slab":
        p = rng.normal(size=(n, 3)) * np.array([1.0, 1.0, 0.25])
        return p
    raise ValueError(shape)


def knn_W(points, k=6):
    """Directed topological k-nearest-neighbour averaging matrix, plus self-loop.

    Row degree is k+1 for EVERY bird (each bird attends to exactly k others), so Q is
    a multiple of the identity even though W is neither symmetric nor doubly stochastic.
    """
    n = len(points)
    d2 = ((points[:, None, :] - points[None, :, :])**2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nb = np.argsort(d2, axis=1)[:, :k]
    W = np.zeros((n, n))
    rows = np.repeat(np.arange(n), k)
    W[rows, nb.ravel()] = 1.0
    W += np.eye(n)
    d = W.sum(1)
    return W / d[:, None], d, nb


# ------------------------------------------------------------------ collisions, counted
def collision_prob_exact(W, T):
    """C_mean(t) = (1/n) sum_i sum_j (W^t)_ij^2, exactly, for t = 0..T."""
    n = W.shape[0]
    P = np.eye(n)
    out = [1.0]
    for t in range(1, T + 1):
        P = P @ W
        out.append(float((P**2).sum() / n))
    return np.array(out)


def collision_prob_walked(nb, T, trials=200000, seed=0):
    """C_mean(t) by WALKING PAIRS OF DRUNKS.  nb is the k-NN table; the lazy step is
    'stay put or move to one of the k neighbours, uniformly'."""
    rng = np.random.default_rng(seed)
    n, k = nb.shape
    start = rng.integers(0, n, trials)
    a = start.copy(); b = start.copy()
    hits = [1.0]
    for t in range(1, T + 1):
        for who in (0, 1):
            arr = a if who == 0 else b
            choice = rng.integers(0, k + 1, trials)          # 0 = stay, 1..k = neighbour
            moved = nb[arr, np.minimum(choice, k) - 1]
            arr[:] = np.where(choice == 0, arr, moved)
        hits.append(float(np.mean(a == b)))
    return np.array(hits)

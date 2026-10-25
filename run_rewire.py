"""R1, R2, R3 -- clustering vs the redundancy tax, at FIXED degree six.

Degree-preserving double-edge swaps on the 6-regular ring: every bird keeps exactly six
neighbours at every stage, so num = w^2 + (1-w)^2 gamma^2/7 never moves and the only
thing changing is the SHAPE of the neighbourhood.
"""
import numpy as np, flock as F, flock3d as F3
from scipy.stats import spearmanr
GAM, W_FIX, N = 0.5, 0.07, 1024

def ring6(n):
    A = np.zeros((n, n), bool)
    for i in range(n):
        for o in (1, 2, 3):
            A[i, (i+o) % n] = A[i, (i-o) % n] = True
    return A

def edges_of(A):
    iu = np.triu_indices(len(A), 1)
    m = A[iu]
    return list(zip(iu[0][m], iu[1][m]))

def swap(A, frac, seed):
    """Degree-preserving double-edge swaps. frac = swaps per edge."""
    A = A.copy(); rng = np.random.default_rng(seed)
    E = edges_of(A); m = len(E)
    target = int(frac * m); done = 0; tries = 0
    while done < target and tries < 80 * max(target, 1):
        tries += 1
        i, j = rng.integers(0, m, 2)
        if i == j: continue
        a, b = E[i]; c, d = E[j]
        if len({a, b, c, d}) < 4: continue
        if rng.random() < 0.5: c, d = d, c
        if A[a, d] or A[c, b]: continue
        A[a, b] = A[b, a] = False; A[c, d] = A[d, c] = False
        A[a, d] = A[d, a] = True;  A[c, b] = A[b, c] = True
        E[i] = (a, d); E[j] = (c, b); done += 1
    return A, done / m

def clustering(A):
    Af = A.astype(float); np.fill_diagonal(Af, 0)
    deg = Af.sum(1); tri = np.diag(Af @ Af @ Af); pairs = deg * (deg - 1)
    return float(np.mean(tri / np.maximum(pairs, 1)))

def spectral_tax(A, w):
    """Exact tax via the eigenvalues of the symmetric averaging matrix (regular graph)."""
    Wm, d = F.W_from_adj(A.astype(float))
    mu = np.linalg.eigvalsh(Wm)
    num = w**2 + (1-w)**2 * GAM**2 / d[0]
    s2 = float(np.mean(num / (1 - (1-w)**2 * mu**2)))
    return s2 / num - 1, s2, d[0], mu

A0 = ring6(N)
print(f"n = {N}, every bird keeps exactly 6 neighbours, gamma = {GAM}, w = {W_FIX}")
print(f"{'swap frac':>10} {'clustering':>11} {'tax':>8} {'n_eff':>8} {'gap 1-mu2':>10}")
rows = []
for frac in (0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 3.0):
    A, got = swap(A0, frac, seed=100 + int(frac*100))
    cl = clustering(A)
    tax, s2, d, mu = spectral_tax(A, W_FIX)
    gap = 1 - np.sort(mu)[-2]
    rows.append((cl, tax))
    print(f"{got:>10.2f} {cl:>11.4f} {tax:>7.1%} {1/s2:>8.2f} {gap:>10.5f}")

cls = np.array([r[0] for r in rows]); tax = np.array([r[1] for r in rows])
rho = spearmanr(cls, tax).statistic
print(f"\nR1  Spearman rho(clustering, tax) = {rho:+.4f}   "
      f"{'HELD' if rho > 0.9 else 'MISSED'}  (predicted > 0.9)")

tax_rand = tax[-1]
print(f"R2  fully rewired (random 6-regular) tax = {tax_rand:.1%}  vs 3-D cubic 21.2%   "
      f"{'HELD' if tax_rand < 0.212 else 'MISSED'}")

# R3: does the real flock land ON this curve?
P = F3.flock_points(1600, seed=3, shape="ball")
Wk, dk, nbk = F3.knn_W(P, k=6)
out = np.zeros((1600, 1600), bool); out[np.repeat(np.arange(1600), 6), nbk.ravel()] = True
cl_f = clustering((out | out.T))
s2_f = F3.sigma2_knn(Wk, dk, W_FIX, GAM)[0].mean()
num_f = W_FIX**2 + (1-W_FIX)**2 * GAM**2 / dk[0]
tax_f = s2_f/num_f - 1
curve = np.interp(cl_f, cls[np.argsort(cls)], tax[np.argsort(cls)])
print(f"R3  real 6-NN flock: clustering {cl_f:.4f}, tax {tax_f:.1%}; "
      f"sweep curve at that clustering predicts {curve:.1%}  "
      f"(off by {abs(tax_f-curve)*100:.1f} pts)")
print(f"    {'HELD (flock is OFF the curve, as predicted)' if abs(tax_f-curve) > 0.02 else 'MISSED -- the flock lands ON the curve; clustering really is the whole of it'}")

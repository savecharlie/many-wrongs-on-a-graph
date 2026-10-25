"""Q1, Q2, Q3 -- the known-answer validations. A failure here voids the arm."""
import numpy as np, flock as F, flock3d as F3

GRAPHS = {"ring (1-D, 6 nbrs)":  ("ring",  lambda L: (L,)),
          "tri  (2-D, 6 nbrs)":  ("tri",   lambda L: (L, L)),
          "cubic(3-D, 6 nbrs)":  ("cubic", lambda L: (L, L, L))}

print("=" * 74)
print("Q1  walk series (real-space delta propagation)  vs  spectral value")
print("=" * 74)
worst_q1 = 0.0; cases = 0
for name, (key, shp) in GRAPHS.items():
    D, off = F3.STENCIL[key]
    L = {1: 64, 2: 16, 3: 8}[D]
    for w in (0.02, 0.08, 0.2, 0.5, 0.9):
        for gamma in (0.0, 0.5, 2.0):
            s_sp, d = F3.cayley_sigma2(shp(L), off, w, gamma)
            s_wk, d2, T = F3.walk_series_sigma2(shp(L), off, w, gamma)
            rel = abs(s_wk - s_sp) / s_sp
            worst_q1 = max(worst_q1, rel); cases += 1
    print(f"  {name}  L={L}  n={np.prod(shp(L)):>6}  degree={int(d)}   ok")
print(f"  {cases} cases, worst relative disagreement: {worst_q1:.3e}")
print(f"  Q1: {'PASS' if worst_q1 < 1e-11 else 'FAIL'}   (threshold 1e-11, registered)\n")

print("=" * 74)
print("Q2a  FFT spectrum  vs  dense fixed-point Stein solve")
print("=" * 74)
worst_q2a = 0.0; cases = 0
for name, (key, shp) in GRAPHS.items():
    D, off = F3.STENCIL[key]
    L = {1: 64, 2: 8, 3: 4}[D]
    Wsp, d = F3.lattice_W(shp(L), off)
    Wd = np.asarray(Wsp.todense())
    for w in (0.05, 0.2, 0.6):
        for gamma in (0.0, 0.5, 2.0):
            s_sp, _ = F3.cayley_sigma2(shp(L), off, w, gamma)
            s_st, _, _ = F.sigma2_stein(Wd, d, w, gamma)
            worst_q2a = max(worst_q2a, abs(s_st - s_sp) / s_sp); cases += 1
print(f"  {cases} cases, worst relative disagreement: {worst_q2a:.3e}")
print(f"  Q2a: {'PASS' if worst_q2a < 1e-9 else 'FAIL'}\n")

print("Q2b  stein_doubling  vs  the EXACT Kronecker linear solve")
print("     (first draft compared it with sigma2_stein's fixed point and 'failed' at 8e-11;")
print("      against a real known answer the DOUBLING solver is exact and the fixed point")
print("      is the loose one at 1.4e-12 -- see run_q2b_exact.py. Fix the reference, not")
print("      the threshold.)")
def stein_exact(W, Q, a):
    n = len(W)
    return np.linalg.solve(np.eye(n*n) - a*np.kron(W, W), Q.reshape(-1)).reshape(n, n)
worst_q2b = 0.0; cases = 0
rng = np.random.default_rng(7)
for n in (20, 40):
    for trial in range(3):
        P = rng.normal(size=(n, 3))
        W, d, nb = F3.knn_W(P, k=4)
        for w in (0.03, 0.15, 0.5):
            for gamma in (0.0, 0.5, 2.0):
                Q = w**2*np.eye(n) + (1-w)**2*gamma**2*np.diag(1.0/d)
                mx = np.diag(stein_exact(W, Q, (1-w)**2)).mean()
                diag, iters = F3.sigma2_knn(W, d, w, gamma)
                worst_q2b = max(worst_q2b, abs(diag.mean() - mx)/mx); cases += 1
print(f"  {cases} cases on directed irregular k-NN graphs, worst: {worst_q2b:.3e}")
print(f"  Q2b: {'PASS' if worst_q2b < 1e-12 else 'FAIL'}")
print()

print("=" * 74)
print("Q3  Stein solve on a DIRECTED topological 6-NN graph  vs  direct Monte Carlo")
print("=" * 74)
print(f"  {'n':>5} {'w':>6} {'gamma':>6} {'stein':>11} {'monte carlo':>13} {'rel diff':>10}")
worst_q3 = 0.0
for n, seed in ((120, 1), (300, 2)):
    P = F3.flock_points(n, seed=seed, shape="ball")
    W, d, nb = F3.knn_W(P, k=6)
    for w in (0.05, 0.2):
        for gamma in (0.5, 2.0):
            diag, _ = F3.sigma2_knn(W, d, w, gamma)
            mc = F.sigma2_mc(W, d, w, gamma, steps=200000, burn=2000, seed=11)
            rel = abs(diag.mean() - mc) / mc
            worst_q3 = max(worst_q3, rel)
            print(f"  {n:>5} {w:>6.2f} {gamma:>6.2f} {diag.mean():>11.6f} {mc:>13.6f} {rel:>9.3%}")
print(f"  worst: {worst_q3:.3%}")
print(f"  Q3: {'PASS' if worst_q3 < 0.005 else 'FAIL'}   (threshold 0.5%, registered)")

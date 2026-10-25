"""Q2b arbitration: both solvers against an EXACT linear solve.

The Stein equation S = Q + a W S W' is linear in S, so vec(S) = (I - a (W kron W))^{-1} vec(Q).
At n=20 that is a 400x400 dense solve -- exact up to LAPACK, no iteration tolerance at all.
This is the known answer; the two iterative solvers are the instruments being checked.
"""
import numpy as np, flock as F, flock3d as F3

def stein_exact(W, Q, a):
    n = len(W)
    M = np.eye(n * n) - a * np.kron(W, W)
    return np.linalg.solve(M, Q.reshape(-1)).reshape(n, n)

rng = np.random.default_rng(7)
print(f"{'n':>4} {'w':>6} {'gam':>5} {'exact mean':>12} {'doubling':>11} {'fixed-pt':>11} "
      f"{'dbl rel':>10} {'fp rel':>10}")
wd = wf = 0.0
for n in (20, 40):
    for trial in range(2):
        P = rng.normal(size=(n, 3))
        W, d, nb = F3.knn_W(P, k=4)
        for w in (0.03, 0.15, 0.5):
            for gamma in (0.5, 2.0):
                Q = w**2*np.eye(n) + (1-w)**2*gamma**2*np.diag(1.0/d)
                a = (1-w)**2
                Sx = stein_exact(W, Q, a);      mx = np.diag(Sx).mean()
                Sd, it = F3.stein_doubling(W, Q, a); md = np.diag(Sd).mean()
                mf, _, _ = F.sigma2_stein(W, d, w, gamma)
                rd, rf = abs(md-mx)/mx, abs(mf-mx)/mx
                wd, wf = max(wd, rd), max(wf, rf)
                print(f"{n:>4} {w:>6.2f} {gamma:>5.1f} {mx:>12.8f} {md:>11.8f} {mf:>11.8f} "
                      f"{rd:>10.1e} {rf:>10.1e}")
print(f"\nworst vs exact:  stein_doubling {wd:.2e}   sigma2_stein(fixed point) {wf:.2e}")
print(f"Q2b verdict: the SOLVER to trust is "
      f"{'stein_doubling' if wd < wf else 'sigma2_stein'}")

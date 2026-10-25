"""Q7 (the surface) and Q8 (the collisions, actually counted) on a real 3-D flock.

The graph here is what Ballerini et al. 2008 measured: TOPOLOGICAL six-nearest-
neighbour, on a finite point cloud with a boundary.  Directed (i may attend to j
without j attending to i) and irregular in IN-degree, though every bird's OUT-degree
is exactly 6, which is what keeps Q a multiple of the identity.
"""
import numpy as np, flock3d as F3
from scipy.optimize import minimize_scalar

GAM, K = 0.5, 6

def neff_at_best_w(W, d, n_lo=1e-4):
    f = lambda w: F3.sigma2_knn(W, d, w, GAM)[0].mean()
    r = minimize_scalar(f, bounds=(n_lo, 0.5), method="bounded", options={"xatol": 1e-6})
    return float(np.sqrt(r.fun)), float(r.x)

print("=" * 80)
print("The real answer: n_eff for a flock on a TOPOLOGICAL 6-nearest-neighbour graph")
print("  (periodic degree-matched lattices gave 19.41 / 21.60 / 23.32 in 1/2/3-D)")
print("=" * 80)
print(f"{'cloud':<22} {'n':>6} {'sigma':>9} {'w*':>8} {'n_eff':>8}")
store = {}
for tag, shape, n, seed in (("isotropic blob", "ball", 800, 3),
                            ("flat slab (starling)", "slab", 800, 4)):
    P = F3.flock_points(n, seed=seed, shape=shape)
    W, d, nb = F3.knn_W(P, k=K)
    s, w = neff_at_best_w(W, d)
    store[tag] = dict(P=P, W=W, d=d, nb=nb, w=w, s=s)
    print(f"{tag:<22} {n:>6} {s:>9.5f} {w:>8.4f} {1/s**2:>8.2f}")

print("\n  saturation check on the blob, at its own w* each time:")
for n in (400, 800, 1600):
    P = F3.flock_points(n, seed=5, shape="ball")
    W, d, nb = F3.knn_W(P, k=K)
    s, w = neff_at_best_w(W, d)
    print(f"    n = {n:>5}   sigma {s:.5f}   w* {w:.4f}   n_eff {1/s**2:.2f}")

print("\n" + "=" * 80)
print("Q7   the surface:  do boundary birds pool worse than core birds?")
print("=" * 80)
for tag in store:
    g = store[tag]
    P, W, d, w = g["P"], g["W"], g["d"], g["w"]
    diag, _ = F3.sigma2_knn(W, d, w, GAM)
    r = np.linalg.norm(P - P.mean(0), axis=1)
    q = np.quantile(r, [0.2, 0.8])
    core, shell = diag[r <= q[0]], diag[r >= q[1]]
    pen = shell.mean() / core.mean() - 1
    print(f"  {tag:<22} core sigma^2 {core.mean():.6f}  shell {shell.mean():.6f}  "
          f"penalty {pen:>+7.2%}")
    # correlation of per-bird variance with radius, as a continuous check
    cc = np.corrcoef(r, diag)[0, 1]
    print(f"  {'':<22} corr(radius, sigma_i^2) = {cc:+.3f}   "
          f"n_eff core {1/core.mean():.2f} vs shell {1/shell.mean():.2f}")
    g["pen"] = pen
pen_ball = store["isotropic blob"]["pen"]
print(f"\n  Q7: {'HELD' if pen_ball >= 0.05 else 'MISSED'}  (predicted shell >= 5% worse; "
      f"blob gives {pen_ball:+.2%})")

print("\n" + "=" * 80)
print("Q8   the collisions, COUNTED:  pairs of drunks walked vs sum_j (W^t)_ij^2")
print("=" * 80)
g = store["isotropic blob"]
exact = F3.collision_prob_exact(g["W"], 8)
walked = F3.collision_prob_walked(g["nb"], 8, trials=400000, seed=1)
print(f"  {'t':>3} {'exact':>12} {'walked':>12} {'diff':>11} {'2 s.e.':>10}")
ok = True
for t in range(1, 9):
    se = np.sqrt(max(walked[t], 1e-9) * (1 - walked[t]) / 400000)
    d_ = abs(walked[t] - exact[t])
    flag = d_ <= 2 * se + 1e-6
    ok &= flag
    print(f"  {t:>3} {exact[t]:>12.6f} {walked[t]:>12.6f} {d_:>11.2e} {2*se:>10.2e}"
          f"  {'ok' if flag else 'OUT'}")
print(f"\n  Q8: {'HELD' if ok else 'MISSED'}  (all t within 2 standard errors)")

print("\n  and the series itself, term by term, at the blob's w* "
      f"(a = (1-w)^2 = {(1-g['w'])**2:.4f}):")
a = (1 - g["w"])**2
num = g["w"]**2 + (1 - g["w"])**2 * GAM**2 / g["d"][0]
tot = 0.0
print(f"  {'t':>3} {'a^t':>9} {'C(t)':>10} {'term':>11} {'cumulative':>12}  share")
full = F3.collision_prob_exact(g["W"], 40)
for t in range(0, 13):
    term = a**t * full[t]
    tot += term
    print(f"  {t:>3} {a**t:>9.5f} {full[t]:>10.6f} {term:>11.6f} {tot:>12.6f}"
          f"  {term/ (num and 1) and term:>.4f}")
allsum = sum(a**t * full[t] for t in range(40))
print(f"  ... total over 40 steps: {allsum:.6f};  num = {num:.6f};  "
      f"sigma^2 = {num*allsum:.6f}  (n_eff {1/(num*allsum):.2f})")
print(f"  the bird's OWN t=0 term is {1/allsum:.1%} of the whole series.")

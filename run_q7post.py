"""Q7 MISSED and reversed. Post hoc, labelled: why are boundary birds BETTER?

Hypothesis after the fact: RECIPROCITY. In the core, pairs of birds are each other's
nearest neighbours, so a walk can bounce i->j->i. A 2-cycle is a collision generator.
At the boundary a bird's six neighbours are all inward and have closer neighbours of
their own, so they do not point back: the local graph is more tree-like, and a tree is
the best possible shape for pooling.
"""
import numpy as np, flock3d as F3
GAM, K = 0.5, 6

print("=" * 78)
print("Post hoc: reciprocity vs radius vs per-bird variance (blob, n=1600)")
print("=" * 78)
for tag, shape, seed in (("blob", "ball", 3), ("slab", "slab", 4)):
    P = F3.flock_points(1600, seed=seed, shape=shape)
    W, d, nb = F3.knn_W(P, k=K)
    n = len(P)
    out = np.zeros((n, n), bool); out[np.repeat(np.arange(n), K), nb.ravel()] = True
    recip = (out & out.T).sum(1) / K                 # fraction of i's out-edges reciprocated
    indeg = out.sum(0)
    w = 0.07
    diag, _ = F3.sigma2_knn(W, d, w, GAM)
    r = np.linalg.norm(P - P.mean(0), axis=1)

    def cc(a, b): return np.corrcoef(a, b)[0, 1]
    def partial(a, b, c):                            # corr(a,b) controlling for c
        ra = a - np.polyval(np.polyfit(c, a, 1), c)
        rb = b - np.polyval(np.polyfit(c, b, 1), c)
        return np.corrcoef(ra, rb)[0, 1]

    print(f"\n  {tag}:  mean reciprocity {recip.mean():.3f}   "
          f"mean in-degree {indeg.mean():.2f} (sd {indeg.std():.2f})")
    print(f"    corr(radius,   sigma_i^2) = {cc(r, diag):+.3f}")
    print(f"    corr(recip,    sigma_i^2) = {cc(recip, diag):+.3f}")
    print(f"    corr(in-degree,sigma_i^2) = {cc(indeg, diag):+.3f}")
    print(f"    corr(radius,   recip)     = {cc(r, recip):+.3f}")
    print(f"    corr(radius, sigma^2 | recip) = {partial(r, diag, recip):+.3f}"
          f"   <- collapses if reciprocity is the whole story")
    print(f"    corr(recip, sigma^2 | radius) = {partial(recip, diag, r):+.3f}")
    q = np.quantile(recip, [0.25, 0.75])
    lo, hi = diag[recip <= q[0]], diag[recip >= q[1]]
    print(f"    low-reciprocity quartile sigma^2 {lo.mean():.6f} (n_eff {1/lo.mean():.2f})  vs"
          f"  high {hi.mean():.6f} (n_eff {1/hi.mean():.2f})   {hi.mean()/lo.mean()-1:+.2%}")

print("\n" + "=" * 78)
print("The redundancy tax:  sigma^2 = num * SUM,  and SUM-1 is the cost of walks returning")
print("  (num is what a perfectly tree-like, redundancy-free pipe would give)")
print("=" * 78)
LAT = [("1-D ring", "ring", (4096,)), ("2-D triangular", "tri", (64, 64)),
       ("3-D cubic", "cubic", (16, 16, 16))]
print(f"{'graph':<22} {'w':>7} {'num':>10} {'sigma^2':>10} {'SUM':>8} {'tax':>8} {'n_eff':>8}")
for w, label in ((None, "own w*"), (0.07, "matched w = 0.07"), (0.02, "matched w = 0.02")):
    print(f"  -- {label} --")
    from scipy.optimize import minimize_scalar
    for name, key, shape in LAT:
        D, off = F3.STENCIL[key]
        if w is None:
            r = minimize_scalar(lambda x: F3.cayley_sigma2(shape, off, x, GAM)[0],
                                bounds=(1e-5, .999), method="bounded",
                                options={"xatol": 1e-11})
            ww = float(r.x)
        else:
            ww = w
        s2, dd = F3.cayley_sigma2(shape, off, ww, GAM)
        num = ww**2 + (1 - ww)**2 * GAM**2 / dd
        print(f"  {name:<20} {ww:>7.4f} {num:>10.6f} {s2:>10.6f} {s2/num:>8.4f} "
              f"{s2/num-1:>7.1%} {1/s2:>8.2f}")
    # and the real flock, same w
    P = F3.flock_points(800, seed=3, shape="ball")
    Wk, dk, nbk = F3.knn_W(P, k=K)
    if w is None:
        r = minimize_scalar(lambda x: F3.sigma2_knn(Wk, dk, x, GAM)[0].mean(),
                            bounds=(1e-4, .5), method="bounded", options={"xatol": 1e-6})
        ww = float(r.x)
    else:
        ww = w
    s2 = F3.sigma2_knn(Wk, dk, ww, GAM)[0].mean()
    num = ww**2 + (1 - ww)**2 * GAM**2 / dk[0]
    print(f"  {'3-D flock, 6-NN':<20} {ww:>7.4f} {num:>10.6f} {s2:>10.6f} {s2/num:>8.4f} "
          f"{s2/num-1:>7.1%} {1/s2:>8.2f}")

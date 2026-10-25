"""Q4, Q5, Q6, Q9 -- the dimension result, on degree-matched periodic lattices.

Six neighbours in every row.  Only the DIMENSION changes.
"""
import numpy as np, flock3d as F3
from scipy.optimize import minimize_scalar

GAM = 0.5
LAT = [("1-D ring",      "ring",  lambda n: (n,)),
       ("2-D triangular", "tri",  lambda n: (int(round(n**0.5)),)*2),
       ("3-D cubic",     "cubic", lambda n: (int(round(n**(1/3))),)*3)]

def best_w(shape, off, gamma=GAM):
    f = lambda w: F3.cayley_sigma2(shape, off, w, gamma)[0]
    r = minimize_scalar(f, bounds=(1e-5, 0.999), method="bounded", options={"xatol": 1e-11})
    return float(np.sqrt(r.fun)), float(r.x)

def ball_size(shape, off, r):
    """|ball of graph radius r|, measured on the actual graph by propagating a delta."""
    W, d = F3.lattice_W(shape, off)
    x = np.zeros(W.shape[0]); x[0] = 1.0
    for _ in range(int(r)):
        x = W @ x
    frac = r - int(r)
    reach = int((x > 1e-18).sum())
    if frac > 0:                        # interpolate between radius floor(r) and floor(r)+1
        x2 = W @ x
        reach = int(round((1 - frac) * reach + frac * (x2 > 1e-18).sum()))
    return reach

print("=" * 86)
print("Q4   n_eff at six neighbours, gamma = 0.5, each lattice at its OWN optimal w")
print("     n = 4096 birds in every row")
print("=" * 86)
print(f"{'interaction graph':<17} {'deg':>4} {'sigma':>9} {'w*':>8} {'n_eff=1/sig^2':>14} "
      f"{'1/w*':>7} {'|ball(1/w*)|':>13} {'ball/n_eff':>11}")
res = {}
for name, key, shp in LAT:
    D, off = F3.STENCIL[key]
    shape = shp(4096)
    n = int(np.prod(shape))
    s, w = best_w(shape, off)
    _, d = F3.cayley_sigma2(shape, off, w, GAM)
    neff = 1 / s**2
    b = ball_size(shape, off, 1.0 / w)
    res[name] = dict(sigma=s, w=w, neff=neff, ball=b, n=n, shape=shape, off=off)
    print(f"{name:<17} {int(d)-1:>4} {s:>9.5f} {w:>8.4f} {neff:>14.2f} "
          f"{1/w:>7.2f} {b:>13d} {b/neff:>11.2f}")

r1, r2, r3 = res["1-D ring"], res["2-D triangular"], res["3-D cubic"]
ordered = r3["neff"] > r2["neff"] > r1["neff"]
ratio = r3["neff"] / r1["neff"]
print(f"\n  ordering 3-D > 2-D > 1-D : {ordered}")
print(f"  n_eff(3-D)/n_eff(1-D)    : {ratio:.3f}   (registered prediction: >= 2)")
print(f"  Q4: {'HELD' if ordered and ratio >= 2 else 'MISSED'}"
      f"{'' if ordered else '  -- ORDERING INVERTED'}"
      f"{'' if ratio >= 2 else f'  -- ordering held but ratio {ratio:.2f} < 2'}")

print("\n" + "=" * 86)
print("Q5   the dimension effect at a MATCHED SMALL w = 0.02 (walk range ~25 steps)")
print("=" * 86)
print(f"{'interaction graph':<17} {'sigma':>9} {'n_eff':>9}")
small = {}
for name, key, shp in LAT:
    D, off = F3.STENCIL[key]
    shape = shp(4096)
    s2, d = F3.cayley_sigma2(shape, off, 0.02, GAM)
    small[name] = 1 / s2
    print(f"{name:<17} {np.sqrt(s2):>9.5f} {1/s2:>9.2f}")
r_small = small["3-D cubic"] / small["1-D ring"]
print(f"\n  3-D/1-D at w=0.02 : {r_small:.3f}")
print(f"  3-D/1-D at own w* : {ratio:.3f}")
print(f"  Q5: {'HELD' if r_small > ratio else 'MISSED'}  (predicted the matched-small-w "
      f"ratio to be LARGER)")

print("\n" + "=" * 86)
print("Q6   does sigma saturate with flock size at fixed six neighbours?  (3-D cubic)")
print("=" * 86)
print(f"{'n':>8} {'L':>5} {'sigma':>10} {'w*':>8} {'n_eff':>9}   1/sqrt(n) would be")
sigs = []
for L in (8, 12, 16, 24, 32, 48, 64):
    shape = (L, L, L); n = L**3
    s, w = best_w(shape, F3.STENCIL["cubic"][1])
    sigs.append(s)
    print(f"{n:>8} {L:>5} {s:>10.5f} {w:>8.4f} {1/s**2:>9.2f}   {1/np.sqrt(n):>10.5f}")
spread = max(sigs[1:]) - min(sigs[1:])          # n = 1728 .. 262144
print(f"\n  spread in sigma over n = 1,728 .. 262,144 : {spread:.2e}")
print(f"  Q6: {'HELD' if spread < 5e-4 else 'MISSED'}  (predicted constant to 3 decimals)")

print("\n" + "=" * 86)
print("Q9   the ball-counting heuristic  n_eff ~= |ball(1/w*)|  -- expected to FAIL in 3-D")
print("=" * 86)
for name in ("1-D ring", "2-D triangular", "3-D cubic"):
    r = res[name]
    over = r["ball"] / r["neff"] - 1
    print(f"  {name:<17} n_eff {r['neff']:>8.2f}   |ball| {r['ball']:>8d}   "
          f"overprediction {over:>+7.1%}")
o3 = res["3-D cubic"]["ball"] / res["3-D cubic"]["neff"] - 1
print(f"\n  Q9: {'HELD (heuristic fails in 3-D as predicted)' if o3 > 0.50 else 'MISSED'}"
      f"   -- 3-D overprediction {o3:+.1%}, predicted > +50%")

import numpy as np, flock as F
from scipy.optimize import minimize_scalar

def mu2(W):
    e = np.sort(np.linalg.eigvalsh(W))[::-1]
    return e[1]

def sig(A, w, gam):
    W, d = F.W_from_adj(A)
    s, _, _ = F.sigma2_stein(W, d, w, gam)
    return np.sqrt(s), W, d

def wstar(A, gam):
    r = minimize_scalar(lambda w: sig(A, w, gam)[0], bounds=(1e-4, 0.999), method="bounded",
                        options={"xatol": 1e-7})
    return r.x, r.fun

n, GAM = 20, 0.5
graphs = {
 "complete (all-to-all)" : F.g_complete(n),
 "circulant {1,2,3,4,10}": F.g_circulant(n, [1,2,3,4,10]),
 "circulant {1,2,3,4}"   : F.g_circulant(n, [1,2,3,4]),
 "circulant {1,5}"       : F.g_circulant(n, [1,5]),
 "ring k=2 {1,2}"        : F.g_circulant(n, [1,2]),
 "ring k=1 {1}"          : F.g_ring(n, 1),
 "two cliques + 1 bridge": F.g_two_cliques(n, 1),
}

print("=== P3 / P4 / P6 : n=20, gamma=0.5 ===")
wc, _ = wstar(graphs["complete (all-to-all)"], GAM)
print(f"complete-graph optimal w* = {wc:.4f}\n")
print(f"{'graph':<24} {'deg':>4} {'1-mu2':>7} {'sigma @w*_cmp':>13} {'x cmp':>6} {'own w*':>7} {'sigma @own':>10}")
rows = []
for name, A in graphs.items():
    s_at_wc, W, d = sig(A, wc, GAM)
    ws, s_own = wstar(A, GAM)
    rows.append((name, d.mean()-1, 1-mu2(W), s_at_wc, ws, s_own))
base = [r for r in rows if r[0].startswith("complete")][0]
for name, deg, gap, s, ws, so in rows:
    print(f"{name:<24} {deg:>4.0f} {gap:>7.4f} {s:>13.5f} {s/base[3]:>5.2f}x {ws:>7.4f} {so:>10.5f}")

ring = [r for r in rows if r[0].startswith("ring k=1")][0]
print(f"\nP3 (ring n=20 vs complete at w*_cmp, predicted >1.5x): "
      f"{ring[3]/base[3]:.2f}x -> {'HELD' if ring[3]/base[3] > 1.5 else 'MISSED'}")

# P4: does the ORDERING track the spectral gap? hold degree fixed to remove the gamma^2/d confound
print("\n=== P4: ordering vs spectral gap, DEGREE HELD FIXED (8-regular circulants, n=20) ===")
eight = {"{1,2,3,4}": [1,2,3,4], "{1,2,3,9}": [1,2,3,9], "{1,2,8,9}": [1,2,8,9],
         "{1,5,7,9}": [1,5,7,9], "{2,4,6,8}": [2,4,6,8], "{1,3,5,7}": [1,3,5,7]}
tab = []
for nm, off in eight.items():
    A = F.g_circulant(n, off)
    if not np.all(A.sum(1) == 8):   # some offsets collide; skip those
        continue
    s, W, d = sig(A, wc, GAM)
    tab.append((nm, 1-mu2(W), s))
tab.sort(key=lambda r: r[1])
print(f"{'offsets':<12} {'1-mu2':>8} {'sigma':>9}")
for nm, gap, s in tab: print(f"{nm:<12} {gap:>8.4f} {s:>9.5f}")
mono = all(tab[i][2] >= tab[i+1][2] - 1e-12 for i in range(len(tab)-1))
print(f"P4 (sigma falls monotonically as the gap grows): {'HELD' if mono else 'MISSED'}")

# P5: clustered vs evenly spread, matched as closely as possible
print("\n=== P5: clustered vs evenly spread (Bode 2012's motif result, re-derived) ===")
cl = F.g_two_cliques(n, 1)
ev = F.g_circulant(n, [1,2,3,4,10])
for nm, A in [("two cliques + 1 bridge", cl), ("circulant {1,2,3,4,10}", ev)]:
    s, W, d = sig(A, wc, GAM)
    print(f"  {nm:<24} edges {int(A.sum()//2):>3}  mean deg {A.sum(1).mean():.1f}  "
          f"1-mu2 {1-mu2(W):.4f}  sigma {s:.5f}")
s_cl = sig(cl, wc, GAM)[0]; s_ev = sig(ev, wc, GAM)[0]
print(f"  clustered / evenly spread = {s_cl/s_ev:.2f}x -> "
      f"{'HELD (clustered worse)' if s_cl > s_ev else 'MISSED'}")

print(f"\nP6 (sparse w* > all-to-all w*): "
      f"ring w*={ring[4]:.4f} vs complete w*={wc:.4f} -> "
      f"{'HELD' if ring[4] > wc else 'MISSED'}")

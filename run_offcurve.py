"""How far off the clustering curve is each graph? (The R3 line only checked the flock.)"""
import numpy as np, flock as F, flock3d as F3
exec(open("fig_walks.py").read().split("# ------------------------------------------------------------------ draw")[0])
curve_c, curve_t = sweep[:,0], sweep[:,1]
print(f"{'graph':<30} {'clustering':>11} {'tax':>8} {'curve says':>11} {'off by':>9} {'gap 1-mu2':>10}")
gaps = {}
for nm, key, shape in (("1-D ring","ring",(1024,)), ("2-D triangular sheet","tri",(32,32)),
                       ("3-D cubic lattice","cubic",(16,16,16))):
    mu_, dd = F3.cayley_spectrum(shape, F3.STENCIL[key][1])
    gaps[nm] = 1 - np.sort(mu_)[-2]
gaps["random 6-regular"] = 1 - np.sort(mur)[-2]
evk = np.linalg.eigvals(Wk); gaps["3-D flock, topological 6-NN"] = 1 - np.sort(evk.real)[-2]
for nm, cl, tx, ne in marks:
    pred = float(np.interp(cl, curve_c, curve_t))
    print(f"{nm:<30} {cl:>11.4f} {tx:>7.1%} {pred:>10.1%} {(tx-pred)*100:>+8.1f}pts {gaps[nm]:>10.4f}")
print("\nThe sweep is built from RINGS, which stay diffusive at long range until they are")
print("rewired. The cubic lattice is triangle-free AND diffusive, a combination the sweep")
print("never visits -- so clustering alone cannot place it.")

"""Two EXACT checks on the post-hoc story, then the clustering comparison.

Check A: the 2-cycle identity.  On a k-NN graph with self-loops and d = k+1,
         (W^2)_ii = (1 + #{j in out(i) : i in out(j)}) / (k+1)^2 = (1 + k*recip_i)/(k+1)^2.
         If that is not exact, my "reciprocity makes 2-cycles" sentence is a story.
Check B: does the number of ways HOME (in-degree) beat radius as a predictor, out of sample?
Then:    clustering coefficient of each graph, to see whether the real flock's extra tax
         over the cubic lattice is short cycles.
"""
import numpy as np, flock3d as F3
GAM, K = 0.5, 6

P = F3.flock_points(1600, seed=3, shape="ball")
W, d, nb = F3.knn_W(P, k=K)
n = len(P)
out = np.zeros((n, n), bool); out[np.repeat(np.arange(n), K), nb.ravel()] = True
recip_cnt = (out & out.T).sum(1)
indeg = out.sum(0)

print("Check A -- the 2-cycle identity")
W2 = W @ W
pred = (1 + recip_cnt) / (K + 1)**2
err = np.max(np.abs(np.diag(W2) - pred))
print(f"  max |(W^2)_ii  -  (1 + k*recip_i)/(k+1)^2|  =  {err:.3e}")
print(f"  {'EXACT' if err < 1e-14 else 'NOT EXACT -- the sentence is wrong'}\n")

print("Check B -- predictors of per-bird variance, same graph, w = 0.07")
diag, _ = F3.sigma2_knn(W, d, 0.07, GAM)
r = np.linalg.norm(P - P.mean(0), axis=1)
ret2 = np.diag(W2)
for nm, x in (("radius", r), ("reciprocated edges", recip_cnt.astype(float)),
              ("in-degree", indeg.astype(float)), ("(W^2)_ii, 2-step return", ret2),
              ("(W^4)_ii, 4-step return", np.diag(W2 @ W2))):
    print(f"  corr({nm:<24}, sigma_i^2) = {np.corrcoef(x, diag)[0,1]:+.3f}")

print("\nClustering: short cycles are collision generators. Fraction of a bird's")
print("neighbour pairs that are themselves neighbours (undirected, self-loop dropped):")
def clustering(A):
    A = (A | A.T).astype(float); np.fill_diagonal(A, 0)
    deg = A.sum(1); tri = np.diag(A @ A @ A)
    pairs = deg * (deg - 1)
    return float(np.mean(np.where(pairs > 0, tri / np.maximum(pairs, 1), 0.0)))
print(f"  3-D flock, topological 6-NN : {clustering(out):.4f}")
for name, key, shape in (("1-D ring, 6 nbrs", "ring", (64,)),
                         ("2-D triangular, 6", "tri", (16, 16)),
                         ("3-D cubic, 6 nbrs", "cubic", (8, 8, 8))):
    Wl, dl = F3.lattice_W(shape, F3.STENCIL[key][1])
    A = np.asarray(Wl.todense()) > 0
    np.fill_diagonal(A, False)
    print(f"  {name:<27} : {clustering(A):.4f}")

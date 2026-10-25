"""The effective number of birds actually being averaged.
A lone bird using only its own signal has sigma = 1 (the signal variance is 1 by
construction in Mann's scaling). So n_eff = 1/sigma^2 is literally "how many
independent compass readings is this bird's heading worth".
POST HOC, labelled: not registered. It is the quantity Ivy's question asks for."""
import numpy as np, flock as F
from scipy.optimize import minimize_scalar
GAM=0.5
def best(A):
    W,d=F.W_from_adj(A)
    r=minimize_scalar(lambda w: F.sigma2_stein(W,d,w,GAM)[0],bounds=(1e-4,.999),
                      method="bounded",options={"xatol":1e-9})
    return np.sqrt(r.fun), r.x
print("n = 256 birds in every row. gamma = 0.5, each at its own optimal w.\n")
print(f"{'interaction graph':<26} {'deg':>4} {'sigma':>8} {'w*':>7} {'n_eff=1/sig^2':>14} {'1/w*':>6} {'ball(1/w*)':>11}")
for nm,A,deg in [("ring, 2 neighbours",F.g_ring(256,1),2),
                 ("circulant, 4",F.g_circulant(256,[1,2]),4),
                 ("circulant, 6",F.g_circulant(256,[1,2,3]),6),
                 ("circulant, 8",F.g_circulant(256,[1,2,3,4]),8),
                 ("circulant, 14",F.g_circulant(256,[1,2,3,4,5,6,7]),14),
                 ("complete (all 255)",F.g_complete(256),255)]:
    s,w=best(A)
    ball=1+deg*(1.0/w)          # birds within 1/w* steps on a deg-regular ring-like graph
    print(f"{nm:<26} {deg:>4} {s:>8.5f} {w:>7.4f} {1/s**2:>14.2f} {1/w:>6.2f} {min(ball,256):>11.1f}")
print("\nSAME, but flock size varied at FIXED degree 8 -- does n_eff move?")
print(f"{'n':>5} {'sigma':>9} {'n_eff':>8}")
for n in (16,32,64,128,256,512):
    s,w=best(F.g_circulant(n,[1,2,3,4]))
    print(f"{n:>5} {s:>9.5f} {1/s**2:>8.2f}")

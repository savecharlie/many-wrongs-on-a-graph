"""POST HOC (labelled): P4 missed. Find out why, and whether a correct invariant exists.
The formula sums 1/(1-(1-w)^2 mu_k^2) over the WHOLE spectrum, and mu is SQUARED,
so eigenvalues near -1 should cost exactly as much as eigenvalues near +1.
The spectral GAP only sees the slowest mode at +1. Hypothesis: the right invariant
is the mean of 1/(1-(1-w)^2 mu^2), i.e. the spectral mass near |mu| = 1."""
import numpy as np, flock as F
n, GAM, W_OPT = 20, 0.5, 0.0541
eight = {"{1,2,3,4}":[1,2,3,4], "{1,2,3,9}":[1,2,3,9], "{1,2,8,9}":[1,2,8,9],
         "{1,5,7,9}":[1,5,7,9], "{1,3,5,7}":[1,3,5,7], "{2,4,6,8}":[2,4,6,8],
         "{1,2,4,8}":[1,2,4,8], "{1,4,6,9}":[1,4,6,9], "{3,4,5,6}":[3,4,5,6]}
print(f"{'offsets':<12} {'conn':>5} {'1-mu2':>7} {'mu_min':>8} {'max|mu|':>8} {'mean 1/(1-a mu^2)':>19} {'sigma':>9}")
rows=[]
for nm,off in eight.items():
    A=F.g_circulant(n,off)
    if not np.all(A.sum(1)==8): continue
    W,d=F.W_from_adj(A)
    mu=np.linalg.eigvalsh(W)
    a=(1-W_OPT)**2
    inv=np.mean(1.0/(1-a*mu**2))
    L=np.diag(A.sum(1))-A
    conn = np.sum(np.abs(np.linalg.eigvalsh(L))<1e-9)==1     # connected iff one zero Laplacian eigenvalue
    s,_,_=F.sigma2_stein(W,d,W_OPT,GAM); s=np.sqrt(s)
    mu_nonmax=np.sort(mu)[::-1][1]
    rows.append((nm,conn,1-mu_nonmax,mu.min(),np.max(np.abs(mu[:-1])),inv,s))
for r in sorted(rows,key=lambda r:r[5]):
    print(f"{r[0]:<12} {str(r[1]):>5} {r[2]:>7.4f} {r[3]:>8.4f} {r[4]:>8.4f} {r[5]:>19.5f} {r[6]:>9.5f}")
conn=[r for r in rows if r[1]]
conn.sort(key=lambda r:r[5])
mono=all(conn[i][6] <= conn[i+1][6]+1e-12 for i in range(len(conn)-1))
print(f"\nconnected graphs only, sorted by mean 1/(1-a mu^2): sigma monotone? {'YES' if mono else 'no'}")
pear=np.corrcoef([r[5] for r in conn],[r[6] for r in conn])[0,1]
print(f"  Pearson r(invariant, sigma) over {len(conn)} connected 8-regular graphs = {pear:.6f}")
gap=np.corrcoef([r[2] for r in conn],[r[6] for r in conn])[0,1]
print(f"  Pearson r(spectral GAP , sigma) for comparison                  = {gap:.6f}")

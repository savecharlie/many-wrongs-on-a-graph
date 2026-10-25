"""The figure: Polya made visible, and the one-number law.

A: collision probability C(t) vs t, log-log. Two independent t-step walks from the same
   bird; C(t) is the chance they land on the same bird. Slope -D/2 is recurrence.
B: the redundancy tax against clustering. The curve is a degree-preserving rewiring sweep
   of the 6-regular ring (every bird keeps exactly six neighbours); the markers are five
   graphs that had nothing to do with building it.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import flock as F, flock3d as F3
from scipy.stats import spearmanr
GAM, W_FIX = 0.5, 0.07
T = 400

def coll_from_spectrum(mu, T):
    """C(t) = (1/n) tr(W^{2t}) for a symmetric regular graph, from its eigenvalues."""
    return np.array([float(np.mean(mu**(2*t))) for t in range(T+1)])

# --- the lattices, big enough that the walk does not wrap in T steps
specs = []
mu, d = F3.cayley_spectrum((40000,), F3.STENCIL["ring"][1]);   specs.append(("1-D ring", mu))
mu, d = F3.cayley_spectrum((500, 500), F3.STENCIL["tri"][1]);  specs.append(("2-D triangular sheet", mu))
mu, d = F3.cayley_spectrum((64, 64, 64), F3.STENCIL["cubic"][1]); specs.append(("3-D cubic lattice", mu))

def ring6(n):
    A = np.zeros((n, n), bool)
    for i in range(n):
        for o in (1, 2, 3): A[i, (i+o) % n] = A[i, (i-o) % n] = True
    return A
def swap(A, frac, seed):
    A = A.copy(); rng = np.random.default_rng(seed)
    iu = np.triu_indices(len(A), 1); m0 = A[iu]
    E = list(zip(iu[0][m0], iu[1][m0])); m = len(E)
    tgt = int(frac*m); done = 0; tries = 0
    while done < tgt and tries < 80*max(tgt,1):
        tries += 1
        i, j = rng.integers(0, m, 2)
        if i == j: continue
        a, b = E[i]; c, e = E[j]
        if len({a,b,c,e}) < 4: continue
        if rng.random() < .5: c, e = e, c
        if A[a,e] or A[c,b]: continue
        A[a,b]=A[b,a]=False; A[c,e]=A[e,c]=False
        A[a,e]=A[e,a]=True;  A[c,b]=A[b,c]=True
        E[i]=(a,e); E[j]=(c,b); done += 1
    return A
def clustering(A):
    Af = A.astype(float); np.fill_diagonal(Af, 0)
    deg = Af.sum(1); tri = np.diag(Af@Af@Af)
    return float(np.mean(tri/np.maximum(deg*(deg-1),1)))
def tax_of(mu, d, w=W_FIX):
    num = w**2 + (1-w)**2*GAM**2/d
    s2 = float(np.mean(num/(1-(1-w)**2*mu**2)))
    return s2/num - 1, s2

Arand = swap(ring6(1024), 3.0, seed=404)
Wr, dr = F.W_from_adj(Arand.astype(float)); mur = np.linalg.eigvalsh(Wr)
specs.append(("random 6-regular", mur))

# --- the real flock
P = F3.flock_points(800, seed=3, shape="ball")
Wk, dk, nbk = F3.knn_W(P, k=6)
C_flock = F3.collision_prob_exact(Wk, 120)

# --- panel B data: the sweep, then the five independent markers
A0 = ring6(1024)
sweep = []
for frac in (0.0,0.01,0.02,0.035,0.05,0.075,0.1,0.15,0.2,0.28,0.35,0.5,0.75,1.0,2.0,3.0):
    A = swap(A0, frac, seed=100+int(frac*1000))
    Wm, dd = F.W_from_adj(A.astype(float))
    mu_ = np.linalg.eigvalsh(Wm)
    t, _ = tax_of(mu_, dd[0])
    sweep.append((clustering(A), t))
sweep = np.array(sorted(sweep))

marks = []
for nm, key, shape in (("1-D ring","ring",(1024,)), ("2-D triangular sheet","tri",(32,32)),
                       ("3-D cubic lattice","cubic",(16,16,16))):
    Wl, dl = F3.lattice_W(shape, F3.STENCIL[key][1])
    A = np.asarray(Wl.todense())>0; np.fill_diagonal(A, False)
    mu_, dd = F3.cayley_spectrum(shape, F3.STENCIL[key][1])
    t, s2 = tax_of(mu_, dd)
    marks.append((nm, clustering(A), t, 1/s2))
out = np.zeros((800,800), bool); out[np.repeat(np.arange(800),6), nbk.ravel()] = True
s2f = F3.sigma2_knn(Wk, dk, W_FIX, GAM)[0].mean()
numf = W_FIX**2 + (1-W_FIX)**2*GAM**2/dk[0]
marks.append(("3-D flock, topological 6-NN", clustering(out|out.T), s2f/numf-1, 1/s2f))
tr, s2r = tax_of(mur, dr[0]); marks.append(("random 6-regular", clustering(Arand), tr, 1/s2r))

# ------------------------------------------------------------------ draw
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,
                     "axes.edgecolor":"#444","axes.labelcolor":"#111",
                     "text.color":"#111","xtick.color":"#444","ytick.color":"#444"})
fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 5.2))
fig.patch.set_facecolor("white")

COL = {"1-D ring":"#c23b22", "2-D triangular sheet":"#d99a00", "3-D cubic lattice":"#2a6fb0",
       "random 6-regular":"#4a4a4a", "3-D flock, topological 6-NN":"#1f8a5b"}
t = np.arange(1, T+1)
for nm, mu_ in specs:
    C = coll_from_spectrum(mu_, T)
    axA.loglog(t, C[1:], lw=1.9, color=COL[nm], label=f"{nm}  (n={len(mu_):,})")
tf = np.arange(1, 121)
axA.loglog(tf, C_flock[1:], lw=2.4, color=COL["3-D flock, topological 6-NN"], ls="--",
           label="3-D flock, topological 6-NN  (n=800)")
for slope, y0 in ((-0.5, 0.145), (-1.0, 0.145), (-1.5, 0.145)):
    axA.loglog(t, y0*t**slope, color="#aaa", lw=0.8, ls=":", zorder=0)
axA.text(250, 0.0115, r"$t^{-1/2}$", color="#c23b22", fontsize=10)
axA.text(105, 1.75e-3, r"$t^{-1}$", color="#d99a00", fontsize=10)
axA.text(150, 1.0e-4, r"$t^{-3/2}$", color="#2a6fb0", fontsize=10)
axA.annotate("flattens at $1/n$:\nthe walk has forgotten\nwhere it started",
             (300, 1.0e-3), (26, 2.9e-5), fontsize=7.6, color="#4a4a4a",
             arrowprops=dict(arrowstyle="->", color="#999", lw=0.8))
axA.set_xlabel("steps $t$ of each walk")
axA.set_ylabel(r"collision probability  $C(t)=\sum_j P_t(i\!\to\! j)^2$")
axA.set_title("A.  two drunks set off from the same bird.\n"
              "how often do they land on the same bird?", fontsize=10.5, loc="left")
axA.legend(frameon=False, fontsize=7.6, loc="lower left")
axA.grid(alpha=.15, which="both")
axA.set_ylim(2e-6, 0.4)

axB.plot(sweep[:,0], sweep[:,1]*100, "-", color="#999", lw=1.5, zorder=1)
axB.plot(sweep[:,0], sweep[:,1]*100, ".", color="#999", ms=5, zorder=1,
         label="rewiring the ring, six neighbours kept throughout")
POS = {"1-D ring":(-0.025, 0.0, "right"),
       "2-D triangular sheet":(0.012, -3.2, "left"),
       "3-D flock, topological 6-NN":(0.016, 2.6, "left"),
       "3-D cubic lattice":(0.0, 5.0, "center"),
       "random 6-regular":(0.028, -2.0, "left")}
for nm, cl, tx, ne in marks:
    axB.plot([cl], [tx*100], "o", ms=10, mfc=COL[nm], mec="white", mew=1.6, zorder=3)
    dx, dy, ha = POS[nm]
    axB.annotate(f"{nm}\n$n_{{eff}}$ = {ne:.1f}", (cl, tx*100), (cl+dx, tx*100+dy),
                 fontsize=8, color=COL[nm], ha=ha, va="center", zorder=4)
cub = [m for m in marks if m[0]=="3-D cubic lattice"][0]
axB.annotate("", (cub[1], cub[2]*100), (cub[1], 17.8), zorder=2,
             arrowprops=dict(arrowstyle="<->", color="#2a6fb0", lw=1.1, ls=(0,(3,2))))
axB.text(0.035, 19.4, "+3.4 pts off the curve:\nno triangles, but still\ndiffusive at long range",
         fontsize=7.6, color="#2a6fb0")
axB.set_xlabel("clustering coefficient   (do a bird's six neighbours watch each other?)")
axB.set_ylabel("redundancy tax   $\\sigma^2/\\mathrm{num}-1$   (%)")
axB.set_title("B.  short cycles set the tax -- until there are none left.\n"
              "six neighbours everywhere, $w$ = 0.07, $\\gamma$ = 0.5",
              fontsize=10.5, loc="left")
axB.legend(frameon=False, fontsize=7.8, loc="upper left")
axB.grid(alpha=.15)
axB.set_xlim(-0.08, 0.78); axB.set_ylim(14, 51)

fig.suptitle("Where the averaging happens: along random walks, and what spoils it is the walks coming back",
             fontsize=12, x=0.010, ha="left", y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("walks_and_clustering.png", dpi=155, facecolor="white")
print("wrote walks_and_clustering.png")
for nm, cl, tx, ne in marks:
    pred = float(np.interp(cl, sweep[:,0], sweep[:,1]))
    print(f"  {nm:<30} cl {cl:.4f}  tax {tx:>6.1%}  curve {pred:>6.1%}  "
          f"off {(tx-pred)*100:>+5.1f}pts  n_eff {ne:>6.2f}")

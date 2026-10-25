"""n_eff at the MEASURED starling interaction number.
Ballerini et al. 2008, PNAS 105:1232, doi 10.1073/pnas.0711437105 -- verified from
the abstract: "each bird interacts on average with a fixed number of neighbors
(six to seven)". Their flocks were "a few thousand members"."""
import numpy as np, flock as F
from scipy.optimize import minimize_scalar

print("=== validate the FFT fast path against the dense Stein solve ===")
worst=0.0
for n in (32,64,100):
    for off in ([1],[1,2],[1,2,3]):
        for w in (0.05,0.2,0.6):
            for gam in (0.0,0.5,2.0):
                s_fft,d,_=F.sigma2_circulant(n,off,w,gam)
                W,dd=F.W_from_adj(F.g_circulant(n,off))
                s_st,_,_=F.sigma2_stein(W,dd,w,gam)
                worst=max(worst,abs(s_fft-s_st)/s_st)
print(f"  81 cases, worst relative disagreement: {worst:.3e}  -> {'PASS' if worst<1e-9 else 'FAIL'}\n")

def best_circ(n,off,gam):
    f=lambda w: F.sigma2_circulant(n,off,w,gam)[0]
    r=minimize_scalar(f,bounds=(1e-5,.999),method="bounded",options={"xatol":1e-10})
    return np.sqrt(r.fun), r.x

N=2048
print(f"=== n_eff for a flock of {N} birds at the measured interaction number ===")
print(f"{'gamma':>6} | {'d=7  sigma':>11} {'w*':>7} {'n_eff':>7} | {'d=13 sigma':>11} {'w*':>7} {'n_eff':>7}")
for gam in (0.0,0.25,0.5,1.0,2.0):
    s6,w6=best_circ(N,[1,2,3],gam)        # self + 6 neighbours = degree 7
    s12,w12=best_circ(N,[1,2,3,4,5,6],gam)
    print(f"{gam:>6.2f} | {s6:>11.5f} {w6:>7.4f} {1/s6**2:>7.2f} | {s12:>11.5f} {w12:>7.4f} {1/s12**2:>7.2f}")

print(f"\n=== flock size is irrelevant at fixed interaction number (6 neighbours, gamma=0.5) ===")
print(f"{'n':>6} {'sigma':>9} {'n_eff':>8}   1/sqrt(n) would be")
for n in (64,256,1024,4096,16384,65536):
    s,w=best_circ(n,[1,2,3],0.5)
    print(f"{n:>6} {s:>9.5f} {1/s**2:>8.2f}   {1/np.sqrt(n):>9.5f}")
s,w=best_circ(65536,[1,2,3],0.5)
print(f"\n65,536 birds, 6 neighbours each: sigma = {s:.5f}, worth {1/s**2:.1f} independent compasses.")
print(f"the 1/sqrt(n) story claims sigma = {1/np.sqrt(65536):.5f}  -- overstated {s/(1/np.sqrt(65536)):.0f}x")

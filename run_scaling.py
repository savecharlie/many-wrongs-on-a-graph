"""Does the many-wrongs 1/sqrt(n) gain survive on a sparse graph?
POST HOC and labelled: not in PREDICTIONS.md. This is the question Ivy actually
asked, and I only saw it was answerable once the closed form existed."""
import numpy as np, flock as F
from scipy.optimize import minimize_scalar
GAM=0.5
def best(A,gam=GAM):
    W,d=F.W_from_adj(A)
    f=lambda w: F.sigma2_stein(W,d,w,gam)[0]
    r=minimize_scalar(f,bounds=(1e-4,0.999),method="bounded",options={"xatol":1e-8})
    return np.sqrt(r.fun), r.x
print("sigma at each graph's OWN optimal w, gamma=0.5")
print(f"{'n':>4} | {'complete':>9} {'w*':>6} | {'ring k=1':>9} {'w*':>6} | {'4-reg {1,2}':>11} {'w*':>6} | {'8-reg':>8} {'w*':>6}")
res={}
for n in (4,8,16,32,64,128,256):
    row=[n]
    sc,wc=best(F.g_complete(n)); row+=[sc,wc]
    sr,wr=best(F.g_ring(n,1));   row+=[sr,wr]
    s4,w4=best(F.g_circulant(n,[1,2])); row+=[s4,w4]
    if n>=10:
        s8,w8=best(F.g_circulant(n,[1,2,3,4])); row+=[s8,w8]
    else:
        row+=[float('nan'),float('nan')]
    res[n]=row
    print(f"{n:>4} | {row[1]:>9.5f} {row[2]:>6.4f} | {row[3]:>9.5f} {row[4]:>6.4f} | "
          f"{row[5]:>11.5f} {row[6]:>6.4f} | {row[7]:>8.5f} {row[8]:>6.4f}")
print("\nratio sigma(n) / sigma(n/2)  -- 1/sqrt(n) scaling would give 0.7071 every row")
ns=sorted(res)
print(f"{'n':>4} {'complete':>9} {'ring':>9} {'4-reg':>9} {'8-reg':>9}")
for a,b in zip(ns,ns[1:]):
    print(f"{b:>4} {res[b][1]/res[a][1]:>9.4f} {res[b][3]/res[a][3]:>9.4f} "
          f"{res[b][5]/res[a][5]:>9.4f} {(res[b][7]/res[a][7]) if res[a][7]==res[a][7] else float('nan'):>9.4f}")
print("\nsigma * sqrt(n)  -- constant means the 1/sqrt(n) many-wrongs law holds")
print(f"{'n':>4} {'complete':>9} {'ring':>9} {'4-reg':>9} {'8-reg':>9}")
for n in ns:
    r=res[n]
    print(f"{n:>4} {r[1]*np.sqrt(n):>9.4f} {r[3]*np.sqrt(n):>9.4f} {r[5]*np.sqrt(n):>9.4f} {r[7]*np.sqrt(n):>9.4f}")

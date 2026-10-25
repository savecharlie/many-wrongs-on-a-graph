import numpy as np, flock as F
print("=== P2: Stein solve vs Monte Carlo, IRREGULAR graphs (spectral form invalid) ===")
def g_path(n):
    A=np.zeros((n,n))
    for i in range(n-1): A[i,i+1]=A[i+1,i]=1
    return A
def g_star(n):
    A=np.zeros((n,n)); A[0,1:]=A[1:,0]=1; return A
def g_rgg(n, r, seed=3):
    rng=np.random.default_rng(seed); P=rng.random((n,2)); A=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            if np.hypot(*(P[i]-P[j]))<r: A[i,j]=A[j,i]=1
    return A
cases=[("path n=12",g_path(12)),("star n=12",g_star(12)),("rgg n=15 r=.45",g_rgg(15,.45))]
print(f"  {'graph':<16} {'degrees':<18} {'stein':>10} {'monte carlo':>12} {'rel diff':>10}")
ok=True
for name,A in cases:
    W,d=F.W_from_adj(A)
    assert np.all(A.sum(1)>0), "disconnected node"
    for w,gam in [(0.3,0.5)]:
        s_st,_,_=F.sigma2_stein(W,d,w,gam)
        s_mc=F.sigma2_mc(W,d,w,gam,steps=300000,seed=11)
        rel=abs(s_st-s_mc)/s_st
        ok &= rel < 0.02
        print(f"  {name:<16} {str(sorted(set(d.astype(int)))):<18} {s_st:>10.5f} {s_mc:>12.5f} {rel:>9.2%}")
print(f"  P2: {'PASS' if ok else 'FAIL'}  (MC noise floor ~1% at 300k steps)")

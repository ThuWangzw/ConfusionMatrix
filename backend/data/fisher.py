import numpy as np


def D(X_prefix, i, j):
    # distance function between [i-1, j-1]
    return np.square(X_prefix[j]-X_prefix[i-1])

def Lq(X_prefix, K):
    N = len(X_prefix)-1
    l = np.zeros((N+1, K+1))
    for n in range(1, N+1):
        l[n, 1] = D(X_prefix, 1, n)
    for k in range(2, K+1):
        for n in range(k, N+1):
            l[n, k] = min([l[j-1, k-1] + D(X_prefix, j, n) for j in range(k, n+1)])
    return l

def B(Lp):
    N = Lp.shape[0] - 1
    K = Lp.shape[1] - 1
    Bl = np.zeros(K+1)
    for k in range(1, K):
        Bl[k] = Lp[N, k] / Lp[N, k+1]
    return Bl


def get_split_pos(X, K=10):
    """
        return the positions to split by Fisher algorithm
    """
    stepsize = 1000
    X = np.array(X)[np.arange(len(X), step=stepsize)]
    N = len(X)
    X_prefix = np.zeros(N+1)
    for i in range(1,N+1):
        X_prefix[i] = X_prefix[i-1] + X[i-1]
    l = Lq(X_prefix, K * 2 + 1)
    Bl = B(l)
    for i in range(1, K):
        if Bl[i]-1 <= 0.05:
            K = i+1
            l = Lq(X_prefix, K * 2 + 1)
            break
    res = []
    for i in range(K, 1, -1):
        dist = np.array([abs(l[t-1, i-1] + D(X_prefix, t, N) - l[N, i]) for t in range(i, N+1)])
        pos = list(range(i, N+1))[int(np.argmin(dist))]
        res.append(pos - 1)
        N = pos-1
    res.reverse()
    res = [i * stepsize for i in res]
    return res
    
    
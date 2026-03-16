import numpy as np

def pq_oetf(L):
    m1 = 2610/16384
    m2 = 2523/32
    c1 = 3424/4096
    c2 = 2413/128
    c3 = 2392/128

    L = np.clip(L / 10000.0, 0, 1)
    num = c1 + c2 * np.power(L, m1)
    den = 1 + c3 * np.power(L, m1)
    return np.power(num / den, m2)

def hlg_oetf(L):
    L = L / 1000.0
    a = 0.17883277
    b = 0.28466892
    c = 0.55991073
    return np.where(L <= 1/12,
                    np.sqrt(3*L),
                    a*np.log(12*L - b) + c)

def bt709_oetf(L):
    L = np.clip(L / 100.0, 0, 1)
    return np.where(L < 0.018,
                    4.5 * L,
                    1.099 * np.power(L, 0.45) - 0.099)
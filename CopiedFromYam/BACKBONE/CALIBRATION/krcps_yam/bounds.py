import numpy as np
from scipy.optimize import brentq
from scipy.stats import binom



def hoeffding_bound(n, delta, loss):
    return (loss + np.sqrt(1 / (2 * n) * np.log(1 / delta))).item()


# def hoeffding_plus(r, loss, n):
#     h1 = lambda u: u * np.log(u / r) + (1 - u) * np.log((1 - u) / (1 - r))
#     return -n * h1(np.maximum(r, loss))

def hoeffding_plus(r, loss, n):
    EPS = 1e-12
    r = np.clip(r, EPS, 1 - EPS)

    def h1(u):
        u = np.clip(u, EPS, 1 - EPS)
        return u * np.log(u / r) + (1 - u) * np.log((1 - u) / (1 - r))

    return -n * h1(np.maximum(r, loss))


def bentkus_plus(r, loss, n):
    return np.log(np.maximum(binom.cdf(np.floor(n * loss), n, r), 1e-10)) + 1


def hoeffding_bentkus_bound(n, delta, loss, maxiter=1000):
    def _tailprob(r):
        hoeffding_mu = hoeffding_plus(r, loss, n)
        bentkus_mu = bentkus_plus(r, loss, n)
        return np.minimum(hoeffding_mu, bentkus_mu) - np.log(delta)

    if _tailprob(1 - 1e-10) > 0:
        return 1
    else:
        try:
            return brentq(_tailprob, loss, 1 - 1e-10, maxiter=maxiter)
        except:
            print(f"BRENTQ RUNTIME ERROR at muhat={loss}")
            return 1.0
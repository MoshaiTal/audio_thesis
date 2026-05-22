from CopiedFromYam.BACKBONE.CALIBRATION.krcps_yam.losses import loss_zero_one
from CopiedFromYam.BACKBONE.CALIBRATION.krcps_yam.bounds import *
import torch
from tqdm import tqdm


def normalize(
    l: torch.Tensor,
    u: torch.Tensor,
    _min: float = 0.0,
    _max: float = 1.0,
    q_eps: float = 1e-06,
) :
    l, u = torch.clamp(l, min=_min, max=_max), torch.clamp(u, min=_min, max=_max)
    l[l <= q_eps] = 0.0
    u[u <= q_eps] = 0.0
    return l, u

def I(_lambda: torch.Tensor,l: torch.Tensor,u: torch.Tensor) :  # changes u l masks according to lamda ( scalar?)
    l_lambda = l - _lambda
    u_lambda = u + _lambda
    return l_lambda,u_lambda #np.clip(l_lambda,0,1), np.clip(u_lambda,0,1)


def rcps( # calculate the rcps protocol in otder to smallest lamda that is being conrtol by the risk on all the cak set.
    rcps_set: torch.Tensor,
    loss_name: str,
    bound_name: str,
    epsilon: float,
    delta: float,
    lambda_max: torch.Tensor,
    stepsize: float,
    eta: torch.Tensor = None,
):
    l,y_hat,u=rcps_set[0],rcps_set[1],rcps_set[2]
    n_rcps = rcps_set.size(1)
    gain=0
    _lambda = lambda_max
    if eta is None:
        eta = torch.ones_like(_lambda)
    l_init,u_init=I(_lambda,l,u)
    loss = loss_zero_one(y_hat, l_init,u_init).mean()
    ucb = hoeffding_bentkus_bound(n_rcps, delta, loss)

    pbar = tqdm(total=epsilon)
    pbar.update(ucb)
    pold = ucb

    while ucb > epsilon:
        _lambda_old = lambda_max
        lambda_max += stepsize * 10
        gain+=stepsize * 10
        _lambda = lambda_max
        l_init, u_init = I(_lambda, l, u)
        loss = loss_zero_one(y_hat, l_init,u_init).mean()
        ucb = hoeffding_bentkus_bound(n_rcps, delta, loss)

    while ucb <= epsilon:
        pbar.update(ucb - pold)
        pold = ucb

        prev_lambda = _lambda.clone()
        if torch.all(prev_lambda == 0):
            break

        _lambda -= stepsize * eta
        gain-=stepsize * eta.unique()

        # _lambda = torch.clamp(_lambda, min=0)

        l_temp,u_temp=I(_lambda, l,u)
        loss = loss_zero_one(y_hat, l_temp, u_temp).mean()
        ucb = hoeffding_bentkus_bound(n_rcps, delta, loss)
    _lambda = prev_lambda
    gain += stepsize * eta.unique()

    pbar.update(epsilon - pold)
    pbar.close()
    return _lambda,gain
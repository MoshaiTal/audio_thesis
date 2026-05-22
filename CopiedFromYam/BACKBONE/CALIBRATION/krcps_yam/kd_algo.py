import os
from typing import Callable, Iterable
from CopiedFromYam.BACKBONE.CALIBRATION.krcps_yam.membership import loss_otsu
import cvxpy as cp
import numpy as np
import torch
from tqdm import tqdm
from time import time
def _gamma_loss_fn(i, offset, q, _lambda):
    i_lambda = i + 2 * _lambda
    inv_i_lambda = cp.multiply(cp.inv_pos(i_lambda), offset)
    loss = 2 * (1 + q) * inv_i_lambda - q
    loss = cp.pos(loss)
    return loss


def _mse_loss_fn(error, interval,_lambda):
    loss = 0.5 * cp.power(error - (interval+2 * _lambda), 2)  # Convex MSE loss
    return loss

def _pk(opt_set, epsilon, pred_set, k, prob_size):
    n_opt = opt_set.size(0)
    opt_l,y_true, opt_u = opt_set[0],opt_set[1],opt_set[2]
    y_pred=pred_set
    error=torch.abs(y_pred-y_true)
    interval = opt_u - opt_l

    k, nk, m = loss_otsu(opt_l, opt_u, k)

    d = np.prod(opt_set.size()[-3:])
    prob_nk = np.round(prob_size / d * nk).astype(int)
    prob_i, prob_j,prob_d, prob_lambda = [], [], [],[]
    start=time()
    for _k, _nk in enumerate(prob_nk):
        _kd, _ki, _kj = torch.nonzero(m[:, :, :, _k] == 1, as_tuple=True)  # Use expanded membership tensor
        _kidx = np.random.choice(
            torch.sum(m[:, :, :, _k]).long().item(), size=_nk, replace=False
        )
        prob_d.extend(_kd[_kidx])
        prob_i.extend(_ki[_kidx])
        prob_j.extend(_kj[_kidx])
        prob_lambda.extend(_nk * [_k])

    _lambda = cp.Variable(k)
    q = cp.Parameter(nonneg=True)

    c = (opt_l + opt_u) / 2
    offset = torch.abs(y_true - c)
    interval_npy, offset_npy,error_npy = interval.numpy(), offset.numpy(),error.numpy()

    # Compute gamma loss across all samples and average
    r_hat = cp.sum(
        _gamma_loss_fn(
            interval_npy[prob_d, prob_i, prob_j],
            offset_npy[prob_d, prob_i, prob_j],
            q,
            _lambda[[prob_lambda]],
        )
    ) / (n_opt * np.sum(prob_nk))

    # obj1 = cp.sum(_mse_loss_fn(interval_npy[prob_d, prob_i, prob_j], error_npy[prob_d, prob_i, prob_j], _lambda[[prob_lambda]] ))
    obj2=cp.sum(cp.multiply(prob_nk, _lambda))
    obj = cp.Minimize(obj2)
    # constraints = [_lambda >= 0, _lambda <= lambda_max, r_hat <= epsilon]
    constraints = [_lambda >= -10, _lambda <= 10,r_hat <= epsilon]

    pk = cp.Problem(obj, constraints)
    end=time()

    return (pk, q, _lambda, obj, obj), m

def calibrate_k_rcps(opt_set, epsilon, pred_set, k, prob_size, gamma: Iterable[float]):
    prob, m = _pk(opt_set, epsilon, pred_set, k, prob_size)
    pk, q, _lambda, obj1, obj2 = prob

    def _solve(gamma, obj1, obj2):
        q.value = gamma / (1 - gamma)

        if os.path.exists(os.path.expanduser("~/mosek/mosek.lic")):
            pk.solve(
                solver=cp.MOSEK,
                verbose=False,
                warm_start=True,
                mosek_params={"MSK_IPAR_NUM_THREADS": 1},
            )
        else:
            pk.solve(verbose=False, warm_start=True)

        # Evaluate obj1 and obj2
        obj1_value = obj1.value
        obj2_value = obj2.value

        # Log or print the values
        # print(f"\nGamma: {gamma}, Obj1: {obj1_value}, Obj2: {obj2_value}, Combined Obj: {pk.value}")

        # Store the results
        lambda_k, obj = torch.tensor(_lambda.value, dtype=torch.float32), pk.value
        return lambda_k, obj, obj1_value, obj2_value

    start = time()
    sol = [_solve(_gamma, obj1, obj2) for _gamma in tqdm(gamma)]
    sol = sorted(sol, key=lambda x: x[1])  # Sort by combined objective (pk.value)
    sol, _, _, _ = zip(*sol)
    lambda_k = sol[0]
    end = time()
    return lambda_k

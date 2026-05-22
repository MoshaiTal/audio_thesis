import numpy as np
import torch
from skimage.filters import threshold_multiotsu
from time import time
from CopiedFromYam.BACKBONE.CALIBRATION.krcps_yam.thresholds import *

def loss_quantile(opt_set, opt_l, opt_u, k):
    loss = np.clip(opt_u - opt_l, 0, 1)

    q = torch.quantile(loss.view(-1), torch.arange(0, 1, 1 / k)[1:]).unique()
    k = len(q) + 1

    m = (k - 1) * torch.ones_like(loss, dtype=torch.long)
    for i, _q in enumerate(reversed(q)):
        m[loss <= _q] = k - (i + 2)

    qcoords = []
    for _k in range(k):
        qcoords.append(torch.nonzero(m == _k, as_tuple=True))

    assert len(qcoords) == len(q) + 1 == k
    assert all([len(_q[0]) == len(_q[1]) for _q in qcoords])
    assert sum([len(_q[0]) for _q in qcoords]) == torch.numel(loss)

    nk = np.empty((k))
    m = torch.zeros(opt_set.size(-2), opt_set.size(-1), k)
    for _k, _q in enumerate(qcoords):
        nk[_k] = len(_q[0])
        m[_q[0], _q[1], _k] = 1
    return k, nk, m


def loss_otsu(opt_l, opt_u, k):
    loss = np.clip(opt_u-opt_l,0,1)
    m_l=[]
    start=time()
    for loss_im_idx in range(loss.shape[0]):
        loss_img=loss[loss_im_idx]
        # t = threshold_multiotsu(loss_img.numpy(), classes=k)
        t=percentile_thresholds(loss_img.numpy(), k)
        k = len(t) + 1

        m = (k - 1) * torch.ones_like(loss_img, dtype=torch.long)
        for i, _t in enumerate(reversed(t)):
            m[loss_img <= _t] = k - (i + 2)

        tcoords = []
        for _k in range(k):
            tcoords.append(torch.nonzero(m == _k, as_tuple=True))

        assert len(tcoords) == len(t) + 1 == k
        assert all([len(_t[0]) == len(_t[1]) for _t in tcoords])
        assert sum([len(_t[0]) for _t in tcoords]) == torch.numel(loss_img)
        if loss_im_idx==0:
            nk = np.zeros((k))
        m = torch.zeros(opt_l.size(-2), opt_l.size(-1), k)
        for _k, _t in enumerate(tcoords):
            nk[_k] += len(_t[0])
            m[_t[0], _t[1], _k] = 1
        m_l.append(m)

    end=time()

    return k, nk, torch.tensor(np.array((m_l)))



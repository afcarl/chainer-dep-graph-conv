
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
import chainer_chemistry


class DependencyGCUpdate(chainer.Chain):
    def __init__(self, dim, n_label):
        super(DependencyGCLayer, self).__init__()
        with self.init_scope():
            # assuming input dim is the same as output dim
            self.arclabel_linears = chainer.ChainList(
                *[chainer_chemistry.links.GraphLinear(dim, dim)
                for _ in range(n_label)])
        self.dim = dim
        self.n_label = n_label

    # TODO: dep_conds should be made in this function?
    def __call__(self, h, adj, deg_conds):
        if self.xp is np:
            zero_array = np.zeros(h.shape, dtype=np.float32)
        else:
            zero_array = self.xp.zeros_like(h)

        fvds = [F.where(cond, h, zero_array) for cond in deg_conds]

        out_h = 0
        for linear, fvd in zip(self.arclabel_linears, fvds):
            out_h = out_h + linear(fvd)
        return out_h


class DependencyGCLayers(chainer.Chain):
    def __init__(self, dim, n_label, n_layer):
        with self.init_scope():
            self.gc_updates = chainer.ChainList(
                *[DependencyGCUpdate(dim, n_label)
                for _ in range(n_layer)])
        self.dim = dim
        self.n_label = n_label
        self.n_layer = n_layer

    def __call__(self, h, adj):
        raise NotImplementedError
        deg_conds = []  # TODO: make deg_conds
        for update in self.gc_updates:
            h = F.relu(update(h, adj, deg_conds))
        return h

# Small multilayer perceptron in numpy
from __future__ import annotations
import numpy as np


class MLP:
    def __init__(self, d_in: int, hidden: int = 32, seed: int = 0, depth: int = 2):
        rng = np.random.default_rng(seed)
        dims = [d_in] + [hidden] * depth + [1]
        self.W = [rng.normal(0, np.sqrt(2.0 / dims[i]), (dims[i], dims[i + 1])) for i in range(len(dims) - 1)]
        self.b = [np.zeros(dims[i + 1]) for i in range(len(dims) - 1)]
        self.dims = dims


    def forward(self, X):
        a = X
        cache = [X]
        for i in range(len(self.W) - 1):
            a = np.maximum(a @ self.W[i] + self.b[i], 0.0)
            cache.append(a)
        out = (a @ self.W[-1] + self.b[-1]).ravel()
        return out, cache

    def predict(self, X):
        return self.forward(np.asarray(X, dtype=np.float64))[0]


    def loss_grad(self, X, y):
        n = len(y)
        out, cache = self.forward(X)
        r = out - y
        loss = float(np.mean(r ** 2))
        gW = [None] * len(self.W)
        gb = [None] * len(self.b)
        g = (2.0 / n) * r[:, None]
        for i in range(len(self.W) - 1, -1, -1):
            a = cache[i]
            gW[i] = a.T @ g
            gb[i] = g.sum(axis=0)
            if i > 0:
                g = (g @ self.W[i].T) * (cache[i] > 0)
        return loss, gW, gb


    def fit(self, X, y, epochs: int = 400, lr: float = 3e-3, batch: int = 64, seed: int = 0, verbose: bool = False, progress=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-8
        self.ymu = y.mean()
        self.ysd = y.std() + 1e-8
        Xs = (X - self.mu) / self.sd
        ys = (y - self.ymu) / self.ysd
        mW = [np.zeros_like(w) for w in self.W]
        vW = [np.zeros_like(w) for w in self.W]
        mb = [np.zeros_like(b) for b in self.b]
        vb = [np.zeros_like(b) for b in self.b]
        rng = np.random.default_rng(seed)
        t = 0
        n = len(ys)
        for ep in range(epochs):
            order = rng.permutation(n)
            for s in range(0, n, batch):
                idx = order[s:s + batch]
                batch_loss, gW, gb = self.loss_grad(Xs[idx], ys[idx])
                t += 1
                for i in range(len(self.W)):
                    mW[i] = 0.9 * mW[i] + 0.1 * gW[i]
                    vW[i] = 0.999 * vW[i] + 0.001 * gW[i] ** 2
                    mb[i] = 0.9 * mb[i] + 0.1 * gb[i]
                    vb[i] = 0.999 * vb[i] + 0.001 * gb[i] ** 2
                    self.W[i] -= lr * (mW[i] / (1 - 0.9 ** t)) / (np.sqrt(vW[i] / (1 - 0.999 ** t)) + 1e-8)
                    self.b[i] -= lr * (mb[i] / (1 - 0.9 ** t)) / (np.sqrt(vb[i] / (1 - 0.999 ** t)) + 1e-8)
            if progress is not None and (ep==0 or (ep+1)%10==0 or ep+1==epochs):
                progress(ep+1,epochs,batch_loss)
            if verbose and ep % 100 == 0:
                pass
        return self

    def predict_raw(self, X):
        Xs = (np.asarray(X, dtype=np.float64) - self.mu) / self.sd
        return self.predict(Xs) * self.ysd + self.ymu

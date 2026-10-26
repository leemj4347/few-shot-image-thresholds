# Predictor training, calibration and evaluation
from pathlib import Path
import os, sys, json, copy, hashlib, time, datetime, argparse, math, zipfile
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = HERE / "core"
sys.path[:0] = [str(SRC), str(SRC)]
import numpy as np, pandas as pd, torch
from torch import nn
from fractions import Fraction
from scipy.stats import binom, t
import task_io as X, registry_evaluation as R
import similarity as SIM, evaluate_registry as EV, criteria as CR
from methods import registry_predictor as F1, conditional_predictor as F, calibration as FC
torch.set_num_threads(1)
CACHE = ROOT / "models"


def dump(p, d):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    X.atomic(p, d)


def sha(p):
    return X.sha(p)


class Predictor(nn.Module):

    def __init__(self, mode, mu, sd, pool="mean"):
        super().__init__()
        self.mode = mode
        self.pool = pool
        self.head = F1.make_head("set_plain", 256) if mode == "set_plain" else F.make_head(mode, F.MODES[mode])
        self.register_buffer("mu", torch.as_tensor(mu, dtype=torch.float32))
        self.register_buffer("sd", torch.as_tensor(sd, dtype=torch.float32))
        self.register_buffer("affine", torch.tensor([0.0, 1.0]))

    def forward(self, x):
        v = (x - self.mu) / self.sd
        if self.mode.endswith("_set") or self.mode == "set_plain":
            e = self.head.embedder(v[:, :, :1])
            h = e.mean(1) if self.pool == "mean" else e.sum(1)
            if self.mode != "set_plain":
                h = torch.cat((h, v[:, 0, 1:]), 1)
            y = self.head.combiner(h).reshape(-1)
        else:
            y = self.head(v).reshape(-1)
        return self.affine[0] + self.affine[1] * y


def encode(d, alpha, mode, domain):
    if mode == "set_plain":
        return F1.encode_input(d["Z_sup"], 100, d["M"], alpha, "set_plain")
    return F.encode_input(d["Z_sup"], 100, d["M"], alpha, mode, domain, d.get("hist_sup"))


def arrays(bs, alpha, mode):
    valid = [b for b in bs if "base" in b.reg and len(b.reg["base"]["Z_sup"]) >= 100 and len(b.reg["base"]["Z_eval"])]
    x = np.stack([encode(b.reg["base"], alpha, mode, b.domain) for b in valid]).astype(np.float32)
    y = np.array([FC.empirical_quantile(b.reg["base"]["Z_eval"], FC.beta_fraction(alpha)) for b in valid], np.float32)
    return (torch.from_numpy(x), torch.from_numpy(y), valid)


def normalize(x, mode):
    if mode.endswith("_set") or mode == "set_plain":
        mu = torch.zeros(x.shape[-1])
        sd = torch.ones(x.shape[-1])
        mu[0] = x[:, :, 0].mean()
        sd[0] = x[:, :, 0].std(unbiased=False)
    else:
        mu = x.mean(0)
        sd = x.std(0, unbiased=False)
    return (mu, torch.where(sd > 0, sd, torch.ones_like(sd)))


def predict(obj, d, a, mode, domain):
    with torch.no_grad():
        return float(obj(torch.tensor(encode(d, a, mode, domain)[None], dtype=torch.float32))[0])


def model(mode, variant, alpha, seed, fit, dev, progress=None):
    path = CACHE / f"{mode}__{variant}__a{alpha}__s{seed}"
    path.mkdir(parents=True, exist_ok=True)
    pt = path / "model.pt"
    jp = path / "metadata.json"
    if not (pt.is_file() and jp.is_file()):
        x, y, fitbs = arrays(fit, alpha, mode)
        xv, yv, devbs = arrays(dev, alpha, mode)
        torch.manual_seed(seed)
        mu, sd = normalize(x, mode) if variant.startswith("normalized") else (torch.zeros(x.shape[-1]), torch.ones(x.shape[-1]))
        obj = Predictor(mode, mu, sd, "sum" if variant.startswith("sum") else "mean")
        epochs = 400 if variant.endswith("400") else 15
        old = None
        reused = False
        if variant == "affine":
            raw, source, _ = model(mode, "raw15", alpha, seed, fit, dev, progress)
            obj = copy.deepcopy(raw)
            with torch.no_grad():
                pp = obj(x).numpy().astype(float)
            yy = y.numpy().astype(float)
            v = pp - pp.mean()
            den = v @ v
            if den <= 0:
                raise ValueError("Affine degeneracy")
            slope = float(v @ (yy - yy.mean()) / den)
            if slope <= 0:
                raise ValueError("Affine not order preserving")
            obj.affine.copy_(torch.tensor([yy.mean() - slope * pp.mean(), slope]))
            history = source["history"]
            best_epoch = source["best_epoch"]
        else:
            opt = torch.optim.Adam(obj.parameters(), lr=0.001)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.1, patience=2, min_lr=1e-06)
            dl = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(x, y),
                batch_size=64,
                shuffle=True,
                generator=torch.Generator().manual_seed(seed),
                num_workers=0,
            )
            best = math.inf
            best_state = None
            best_epoch = 0
            history = []
            for epoch in range(1, epochs + 1):
                obj.train()
                total = 0
                lr = opt.param_groups[0]["lr"]
                for xb, yb in dl:
                    opt.zero_grad()
                    loss = ((obj(xb) - yb) ** 2).mean()
                    assert torch.isfinite(loss)
                    loss.backward()
                    opt.step()
                    total += float(loss.detach()) * len(yb)
                obj.eval()
                with torch.no_grad():
                    vl = float(((obj(xv) - yv) ** 2).mean())
                history.append(dict(epoch=epoch, train_mse=total / len(y), validation_mse=vl, lr=lr))
                if vl < best:
                    best = vl
                    best_epoch = epoch
                    best_state = copy.deepcopy(obj.state_dict())
                sched.step(vl)
            obj.load_state_dict(best_state)
        obj.eval()
        with torch.no_grad():
            vl = float(((obj(xv) - yv) ** 2).mean())
        md = dict(
            mode=mode,
            variant=variant,
            alpha=alpha,
            beta=round(1 - alpha, 2),
            training_seed=seed,
            epochs=epochs,
            best_epoch=best_epoch,
            development_mse=vl,
            history=history,
            mu=obj.mu.tolist(),
            sd=obj.sd.tolist(),
            pool=obj.pool,
            parameter_count=sum((p.numel() for p in obj.parameters())),
            reused=reused,
            fit_keys=[list(F.key(b)) for b in fitbs],
            validation_keys=[list(F.key(b)) for b in devbs],
            source=str(old) if old else "new fit",
        )
        torch.save(obj.state_dict(), pt)
        dump(jp, md)
    md = json.loads(jp.read_text())
    torch.manual_seed(seed)
    obj = Predictor(mode, md["mu"], md["sd"], md["pool"])
    obj.load_state_dict(torch.load(pt, map_location="cpu", weights_only=True))
    obj.eval()
    return (obj, md, path)


def residual_cal(rows, alpha, kind="np"):
    rr = sorted((Fraction.from_float(FC.empirical_quantile(z, FC.beta_fraction(alpha))) - Fraction.from_float(q) for q, z in rows))
    L = len(rr)
    valid = [k for k in range(1, L + 1) if binom.sf(k - 1, L, 0.95) <= 0.05]
    rank = (min(valid) if valid else L + 1) if kind == "np" else math.ceil((L + 1) * 0.95)
    if rank > L:
        return dict(L=L, rank=rank, Lambda=math.inf, delta=0.05, eta=0.05)
    return dict(
        L=L,
        rank=rank,
        Lambda=FC._ceil_float(rr[rank - 1]),
        lambda_exact=str(rr[rank - 1]),
        delta=0.05,
        eta=0.05,
    )


def calibration(obj, mode, a, cal, kind):
    groups = {}
    for b in cal:
        d = b.reg["base"]
        q = predict(obj, d, a, mode, b.domain)
        groups.setdefault(b.domain if kind.endswith("domain") else "global", []).append((q, d["Z_eval"]))
    result = {}
    for g, rows in groups.items():
        result[g] = FC.calibrate_predictions(rows, a) if kind.startswith("fisch") else residual_cal(
            rows,
            a,
            "conformal" if kind.startswith("conformal") else "np",
        )
        if kind.startswith("fisch"):
            assert result[g].get("constraint_verified", False)
    return result

# Conditional set and summary quantile predictors
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from . import calibration as FC
from . import statmlp as SUMMARY
import similarity as SIM
HIDDEN = 256
EPOCHS = 15
SOURCE = Path(__file__).resolve().parents[3] / "external/fisch/mini_imagenet/quantile_snn.py"
HASH_FILE = Path(__file__).resolve().parents[3] / "external/fisch/source_hashes.json"
COMMIT = "bcd18d1726541da827f7530a724460bd735bc0ed"


# Load the DeepSet class from the author source
def source_class():
    raw = SOURCE.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    record = json.loads(HASH_FILE.read_text(encoding="utf-8-sig"))
    expected = next(x["sha256"].lower() for x in record if x["file"] == "mini_imagenet/quantile_snn.py")
    if digest != expected:
        raise RuntimeError("Author source SHA256 mismatch")
    tree = ast.parse(raw.decode("utf-8"))
    node = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DeepSet"))
    scope = {"nn": nn}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), scope)
    return (scope["DeepSet"], digest)
DOMAINS = ("disc21", "mixed", "solana")
DOMAIN_LABELS = DOMAINS + ("nft",)
INPUTS = {"base": 1, "domain": 1 + len(DOMAIN_LABELS), "pair": 8}


def count_parameters(arch, hidden, c):
    return 3 * hidden * hidden + (6 + c) * hidden + 1 if arch == "set" else hidden * hidden + (10 + c) * hidden + 1


def matched_width(c):
    target = count_parameters("set", 256, c)
    return min(range(1, 1001), key=lambda h: (abs(count_parameters("summary", h, c) - target), h))
MODES = {f"{v}_{a}": 256 if a in ("set", "summary256") else matched_width(c) for v, c in INPUTS.items() for a in ("set", "summary256", "summarymatched")}
PARAMETERS = {m: count_parameters("set" if m.endswith("_set") else "summary", h, INPUTS[m.split("_")[0]]) for m, h in MODES.items()}
GRID = SIM.score_grid(SIM.SCHEME_W["dbp"]).ravel()
GRID_ORDER = np.argsort(GRID, kind="stable")


def pair_features(hist, n, M):
    if not isinstance(hist, dict) or n not in hist:
        raise ValueError("Missing support pair histogram")
    H = np.asarray(hist[n])
    if H.shape != (65, 65) or not np.isfinite(H).all() or (H < 0).any() or (not np.equal(H, np.floor(H)).all()):
        raise ValueError("Invalid integer pair histogram")
    N = int(H.sum())
    if N != n * int(M) or N <= 0:
        raise ValueError("Pair histogram information budget mismatch")
    counts = H.ravel()[GRID_ORDER].astype(np.int64)
    vals = GRID[GRID_ORDER]
    mask = counts > 0
    counts = counts[mask]
    vals = vals[mask]
    cdf = np.cumsum(counts)

    def ascending(rank):
        return float(vals[np.searchsorted(cdf, rank, side="left")])

    def descending(rank):
        return ascending(N - min(rank, N) + 1)
    k = max(1, N // 8)
    p = counts.astype(np.float64) / N
    mean = float(p @ vals)
    variance = float(p @ (vals - mean) ** 2)
    std = np.sqrt(max(variance, 0.0))
    skew = float(p @ (vals - mean) ** 3 / std ** 3) if std > 1e-12 else 0.0
    median = ascending((N + 1) // 2) if N % 2 else (ascending(N // 2) + ascending(N // 2 + 1)) / 2
    return np.array([descending(k), descending(2 * k), descending(4 * k), float(vals[-1]), median, std, skew])


def feature_names(mode):
    variant, arch = mode.split("_", 1)
    names = ["log10M"]
    if variant == "domain":
        names += ["domain_" + d for d in DOMAIN_LABELS]
    if variant == "pair":
        names += ["pair_" + x for x in SUMMARY.FEATURES[:7]]
    return (["score_set"] if arch == "set" else SUMMARY.FEATURES[:7]) + names


def context(mode, n, M, domain, hist):
    variant = mode.split("_")[0]
    v = [np.log10(M)]
    if variant == "domain":
        if domain not in DOMAIN_LABELS:
            raise ValueError("Unknown domain")
        v.extend(float(domain == d) for d in DOMAIN_LABELS)
    if variant == "pair":
        v.extend(pair_features(hist, n, M))
    return np.asarray(v, dtype=np.float64)


class ConditionalSet(nn.Module):

    def __init__(self, head, c):
        super().__init__()
        self.c = c
        self.embedder = head.embedder
        self.combiner = copy.deepcopy(head.combiner)
        old = self.combiner[0]
        layer = nn.Linear(old.in_features + c, old.out_features)
        with torch.no_grad():
            layer.weight[:, :old.in_features].copy_(old.weight)
            layer.weight[:, old.in_features:].zero_()
            layer.bias.copy_(old.bias)
        self.combiner[0] = layer

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != 1 + self.c:
            raise ValueError("Conditional tensor shape mismatch")
        if not torch.equal(x[:, :, 1:], x[:, 0:1, 1:].expand_as(x[:, :, 1:])):
            raise ValueError("Task context not constant")
        h = self.embedder(x[:, :, :1]).mean(1)
        return self.combiner(torch.cat((h, x[:, 0, 1:]), 1)).view(-1)


def make_head(mode, hidden):
    if mode not in MODES or hidden != MODES[mode]:
        raise ValueError("Model width differs from prior specification")
    variant, arch = mode.split("_", 1)
    c = INPUTS[variant]
    cls, _ = source_class()
    head = cls(argparse.Namespace(set_hidden_size=int(hidden)))
    if arch == "set":
        out = ConditionalSet(head, c)
    else:
        out = copy.deepcopy(head.combiner)
        base = nn.Linear(8, int(hidden))
        if c == 1:
            out[0] = base
        else:
            layer = nn.Linear(7 + c, int(hidden))
            with torch.no_grad():
                layer.weight[:, :8].copy_(base.weight)
                layer.weight[:, 8:].zero_()
                layer.bias.copy_(base.bias)
            out[0] = layer
    if sum((p.numel() for p in out.parameters())) != PARAMETERS[mode]:
        raise ValueError("Parameter count mismatch")
    return out


def encode_input(z, n, M, alpha, mode, domain=None, hist=None):
    z = np.asarray(z, dtype=np.float64)
    if len(z) < n or n != 100 or (not np.isfinite(z[:n]).all()):
        raise ValueError("Require 100 normal query scores")
    if not np.isfinite(M) or M < 1 or int(M) != M:
        raise ValueError("Invalid actual M")
    c = context(mode, n, M, domain, hist)
    if mode.endswith("_set"):
        return np.column_stack((z[:n], np.tile(c, (n, 1))))
    return np.concatenate((SUMMARY.features(z, n, M, alpha)[:7], c))


def key(b):
    return (b.scheme, b.role, b.domain, b.structure, int(b.M), int(b.seed))


def validate_rows(bundles, n, kind="base"):
    out = []
    for b in bundles:
        d = b.reg.get(kind)
        if d is None:
            continue
        s = np.asarray(d["Z_sup"], dtype=np.float64)
        z = np.asarray(d["Z_eval"], dtype=np.float64)
        if len(s) < n or not len(z):
            continue
        if not np.isfinite(s[:n]).all() or not np.isfinite(z).all():
            raise ValueError("Nonfinite support/extra scores")
        out.append(b)
    if not out:
        raise ValueError("No valid task rows")
    if len({key(b) for b in out}) != len(out):
        raise ValueError("Duplicate task keys")
    return out


def arrays(bundles, alpha, n, mode):
    bs = validate_rows(bundles, n)
    x = []
    y = []
    for b in bs:
        d = b.reg["base"]
        row = encode_input(d["Z_sup"], n, d["M"], alpha, mode, b.domain, d.get("hist_sup"))
        x.append(row)
        y.append(FC.empirical_quantile(d["Z_eval"], FC.beta_fraction(alpha)))
    x = np.asarray(x, dtype=np.float32)
    return (torch.from_numpy(x), torch.tensor(y, dtype=torch.float32), bs)


@dataclass
class Family:
    models: dict
    mode: str
    metadata: dict


# Train and evaluate a quantile predictor
class Backend:
    HIDDEN = HIDDEN
    EPOCHS = EPOCHS

    def __init__(self, mode, validation_bundles, output):
        self.mode = mode
        self.validation = list(validation_bundles)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)

    def fit(self, train_bundles, alphas, n_list, seed=42, hidden=HIDDEN, epochs=EPOCHS, kind="base", progress=None):
        if kind != "base":
            raise ValueError("Training requires base tasks")
        if not self.validation:
            raise ValueError("Development tasks required")
        if any(b.role != "train" for b in train_bundles):
            raise ValueError("Fit role must be train")
        if any(b.role != "dev" for b in self.validation):
            raise ValueError("Validation role must be dev")
        if set(map(key, train_bundles)) & set(map(key, self.validation)):
            raise ValueError("Fit/validation task overlap")
        if int(epochs) < 1:
            raise ValueError("Positive epoch count required")
        models = {}
        record = {
            "mode": self.mode,
            "source_commit": COMMIT,
            "epochs": int(epochs),
            "hidden": int(hidden),
            "lr": 0.001,
            "batch_size": 64,
            "pooling": "author mean",
            "standardization": False,
            "source_file": str(SOURCE),
            "families": {},
            "features": feature_names(self.mode),
            "initialization": "untrained author source; conditional extra column zero",
        }
        for a in alphas:
            for n in n_list:
                torch.manual_seed(int(seed))
                torch.set_num_threads(1)
                x, y, fitb = arrays(train_bundles, a, n, self.mode)
                xv, yv, valb = arrays(self.validation, a, n, self.mode)
                model = make_head(self.mode, hidden)
                optim = torch.optim.Adam(model.parameters(), lr=0.001)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.1, patience=2, min_lr=1e-06)
                loader = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(x, y),
                    batch_size=64,
                    shuffle=True,
                    num_workers=0,
                    generator=torch.Generator().manual_seed(int(seed)),
                )
                history = []
                best = float("inf")
                best_state = None
                best_epoch = 0
                updates = 0
                for e in range(1, int(epochs) + 1):
                    model.train()
                    total = 0.0
                    count = 0
                    lr = float(optim.param_groups[0]["lr"])
                    for xb, yb in loader:
                        optim.zero_grad()
                        pred = model(xb).reshape(-1)
                        loss = (yb - pred).pow(2).mean()
                        if not torch.isfinite(loss):
                            raise RuntimeError("Nonfinite training loss")
                        loss.backward()
                        optim.step()
                        updates += 1
                        total += float(loss.detach()) * len(yb)
                        count += len(yb)
                    model.eval()
                    with torch.no_grad():
                        vl = float((yv - model(xv).reshape(-1)).pow(2).mean())
                    if not np.isfinite(vl):
                        raise RuntimeError("Nonfinite validation loss")
                    if vl < best:
                        best = vl
                        best_state = copy.deepcopy(model.state_dict())
                        best_epoch = e
                    scheduler.step(vl)
                    row = {
                        "epoch": e,
                        "train_mse": total / count,
                        "validation_mse": vl,
                        "learning_rate_used": lr,
                        "optimizer_updates": updates,
                    }
                    history.append(row)
                    pass
                    if progress:
                        progress(e, int(epochs), total / count)
                    if getattr(self, "overall", None) is not None:
                        self.overall.advance(
                            "training",
                            model=self.mode,
                            alpha=float(a),
                            n=int(n),
                            epoch=e,
                            epochs=int(epochs),
                        )
                model.load_state_dict(best_state)
                model.eval()
                label = f"alpha_{a}_n_{n}"
                torch.save(model.state_dict(), self.output / (label + ".pt"))
                detail = {
                    "fit_task_keys": [list(key(b)) for b in fitb],
                    "validation_task_keys": [list(key(b)) for b in valb],
                    "n_train": len(fitb),
                    "n_validation": len(valb),
                    "best_epoch": best_epoch,
                    "best_validation_mse": best,
                    "optimizer_updates": updates,
                    "parameter_count": sum((p.numel() for p in model.parameters())),
                    "history": history,
                }
                (self.output / (label + ".json")).write_text(json.dumps(detail, indent=2), encoding="utf-8")
                models[float(a), int(n)] = model
                record["families"][label] = detail
        _, record["source_sha256"] = source_class()
        (self.output / "training_manifest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        return (Family(models, self.mode, record), record)

    def predict_raw(self, family, Z_sup, n, M, alpha, domain=None, hist=None):
        z = np.asarray(Z_sup, dtype=np.float64)
        if len(z) < n or n < 1 or (not np.isfinite(z[:n]).all()):
            raise ValueError("Invalid support")
        x = encode_input(z, n, M, alpha, family.mode, domain, hist)[None, ...]
        with torch.no_grad():
            q = float(family.models[float(alpha), int(n)](torch.tensor(x, dtype=torch.float32)).reshape(-1)[0])
        if not np.isfinite(q):
            raise RuntimeError("Nonfinite prediction")
        return q

    def calibrate(self, family, cal_bundles, alpha, n, kind="base", **unused):
        fit_keys = set((tuple(x) for r in family.metadata["families"].values() for x in r["fit_task_keys"]))
        val_keys = set((tuple(x) for r in family.metadata["families"].values() for x in r["validation_task_keys"]))
        if any((key(b) in fit_keys | val_keys for b in cal_bundles)):
            raise ValueError("Calibration task overlap")
        if any(b.role != "train" for b in cal_bundles):
            raise ValueError("Calibration role must be train")
        rows = []
        for b in cal_bundles:
            d = b.reg.get(kind)
            if d is None or len(d["Z_sup"]) < n or (not len(d["Z_eval"])):
                continue
            rows.append((self.predict_raw(family, d["Z_sup"], n, d["M"], alpha, b.domain, d.get("hist_sup")), d["Z_eval"]))
        record = FC.calibrate_predictions(rows, alpha)
        record["calibration_task_keys"] = [list(key(b)) for b in cal_bundles]
        return record

    def tau(self, family, lam, Z_sup, n, M, alpha, w, domain=None, hist=None):
        q = self.predict_raw(family, Z_sup, n, M, alpha, domain, hist)
        return SIM.strict_successor(FC.corrected_boundary(q, lam), w)

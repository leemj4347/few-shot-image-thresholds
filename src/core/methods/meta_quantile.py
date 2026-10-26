# DeepSet quantile prediction for registry scores
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


def make_head(mode="deepset", hidden=HIDDEN):
    cls, _ = source_class()
    head = cls(argparse.Namespace(set_hidden_size=int(hidden)))
    if mode == "deepset":
        return head
    if mode != "summary":
        raise ValueError("unknown model mode")
    combiner = copy.deepcopy(head.combiner)
    combiner[0] = nn.Linear(len(SUMMARY.FEATURES), int(hidden))
    return combiner


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
        row = d["Z_sup"][:n] if mode == "deepset" else SUMMARY.features(d["Z_sup"], n, d["M"], alpha)
        x.append(row)
        y.append(FC.empirical_quantile(d["Z_eval"], FC.beta_fraction(alpha)))
    x = np.asarray(x, dtype=np.float32)
    if mode == "deepset":
        x = x[..., None]
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

    def predict_raw(self, family, Z_sup, n, M, alpha):
        z = np.asarray(Z_sup, dtype=np.float64)
        if len(z) < n or n < 1 or (not np.isfinite(z[:n]).all()):
            raise ValueError("Invalid support")
        if family.mode == "deepset":
            x = z[:n][None, :, None]
        else:
            x = SUMMARY.features(z, n, M, alpha)[None, :]
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
            rows.append((self.predict_raw(family, d["Z_sup"], n, d["M"], alpha), d["Z_eval"]))
        return FC.calibrate_predictions(rows, alpha)

    def tau(self, family, lam, Z_sup, n, M, alpha, w):
        q = self.predict_raw(family, Z_sup, n, M, alpha)
        return SIM.strict_successor(FC.corrected_boundary(q, lam), w)

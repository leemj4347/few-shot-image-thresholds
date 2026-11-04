# Conditional set and summary quantile predictors
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .author_source import load_deepset
from . import statmlp as SUMMARY
import similarity as SIM
HIDDEN = 256
EPOCHS = 15
SOURCE = Path(__file__).resolve().parents[3] / "external/fisch/mini_imagenet/quantile_snn.py"
HASH_FILE = Path(__file__).resolve().parents[3] / "external/fisch/source_hashes.json"
COMMIT = "bcd18d1726541da827f7530a724460bd735bc0ed"


# Load the DeepSet class from the author source
def source_class():
    return load_deepset(SOURCE, HASH_FILE)
INPUTS = {"base": 1}
MODES = {"base_set": 256, "base_summary256": 256}
PARAMETERS = {"base_set": 198401, "base_summary256": 68353}


def feature_names(mode):
    return (["score_set"] if mode.endswith("_set") else SUMMARY.FEATURES[:7]) + ["log10M"]


def context(mode, n, M, domain, hist):
    if mode not in MODES:
        raise ValueError(f"Unknown model mode: {mode}")
    return np.asarray([np.log10(M)], dtype=np.float64)


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
        raise ValueError(f"Expected hidden width {MODES.get(mode)} for mode {mode!r}; got {hidden}")
    variant, arch = mode.split("_", 1)
    c = INPUTS[variant]
    cls, _ = source_class()
    head = cls(argparse.Namespace(set_hidden_size=int(hidden)))
    if arch == "set":
        out = ConditionalSet(head, c)
    else:
        out = copy.deepcopy(head.combiner)
        base = nn.Linear(8, int(hidden))
        out[0] = base
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

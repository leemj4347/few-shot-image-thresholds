from pathlib import Path
import ast
import hashlib
import json
from torch import nn


def load_deepset(source, hashes):
    source = Path(source)
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    records = json.loads(Path(hashes).read_text(encoding="utf-8-sig"))
    expected = next(r["sha256"].lower() for r in records if r["file"] == "mini_imagenet/quantile_snn.py")
    if digest != expected:
        raise ValueError("Author source SHA256 mismatch")
    tree = ast.parse(raw.decode("utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DeepSet")
    scope = {"nn": nn}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
    return scope["DeepSet"], digest

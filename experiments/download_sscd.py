# Download and check the SSCD weights
from __future__ import annotations
import urllib.request, shutil, datetime
import numpy as np
import experiment as C


def main(args, out, holder):
    p = C.Progress("sscd_download", "SSCD 가중치 받기·확인", {"download": 1, "check": 1}, args.smoke, args.log)
    holder["progress"] = p
    C.WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    src = "existing file"
    if not C.WEIGHTS.exists():
        last = None
        for url in C.WEIGHT_URLS:
            try:
                p.stage("download", url=url)
                tmp = C.output_path(C.WEIGHTS.with_suffix(".part"))
                with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f: shutil.copyfileobj(r, f, 1 << 20)
                tmp.replace(C.WEIGHTS)
                src = url
                break
            except Exception as e:
                last = e
        if not C.WEIGHTS.exists(): raise RuntimeError(f"download failed from all sources: {last!r}")
    p.advance("download", source=src)
    digest = C.sha256(C.WEIGHTS)
    model = C.sscd_load()
    img = (np.random.default_rng(0).random((300, 400, 3)) * 255).astype(np.uint8)
    v = C.sscd_embed(model, img)
    rt = C.sscd_runtime_info()
    info = {"file": str(C.WEIGHTS), "source": src, "sha256": digest, "bytes": C.WEIGHTS.stat().st_size,
            "dim": int(v.shape[0]), "norm": float(np.linalg.norm(v)), "time": datetime.datetime.now().isoformat(),
            "preprocess": "RGB, short side 288 (bilinear, aspect kept), ImageNet mean/std; output L2-normalized",
            "reference": "Pizzi et al., A Self-Supervised Descriptor for Image Copy Detection, CVPR 2022",
            "runtime": rt}


    canonical = C.WEIGHTS.parent / "weights_info.json"
    if C.RUN_ROOT is None or not canonical.exists(): C.atomic_json(canonical, info)
    p.advance("check", dim=info["dim"], device=rt["device"])
    return info


if __name__ == "__main__":
    C.run_step("sscd_download", "SSCD 가중치 받기·확인", main)

from __future__ import annotations
import os, sys, json, time, hashlib, argparse
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent
_CFG_FILE = ROOT / 'configs' / 'paths.json'
PATHS_CFG = json.loads(_CFG_FILE.read_text(encoding='utf-8-sig')) if _CFG_FILE.exists() else {}
RUN_ROOT = Path(os.environ['RUN_ROOT']).resolve() if os.environ.get('RUN_ROOT') else None
if RUN_ROOT is not None and ROOT / 'runs' not in RUN_ROOT.parents:
    raise ValueError(f"RUN_ROOT must be a folder inside {ROOT / 'runs'}: {RUN_ROOT}")
OUT_ROOT = RUN_ROOT or ROOT
MODULE_DIR = ROOT / 'src'
ARMS = OUT_ROOT / 'results' / 'main' / 'arms'
MODELS = ROOT / 'models'
RESULTS = OUT_ROOT / 'results'
BASE_RESULTS = ROOT / 'results'

def stored(p):
    p = Path(p)
    if RUN_ROOT is None or p.exists():
        return p
    try:
        rel = p.relative_to(RESULTS)
    except ValueError:
        return p
    return BASE_RESULTS / rel
REFERENCE = ROOT / 'results' / 'tasks' / 'reference'
INPUT_CACHE = ROOT / 'data' / 'input_cache'
INPUTS = ROOT / 'data' / 'inputs'
FISCH_SOURCE = ROOT / 'external' / 'fisch' / 'mini_imagenet' / 'quantile_snn.py'
FISCH_HASH = ROOT / 'external' / 'fisch' / 'source_hashes.json'
SEEDS = (42, 43, 44, 45, 46)
VAR = {'base_set': 'normalized15', 'base_summary256': 'raw15'}
METHODS = {'base_set': 'Meta set', 'base_summary256': 'Summary H256'}
K = ['domain', 'structure', 'kind', 'M', 'scheme', 'n', 'alpha', 'beta', 'method', 'training_seed', 'boundary_case']
TASK_KEY = ['domain', 'structure', 'kind', 'M', 'seed']
DELTA = 0.05
EDIT_GROUPS = {'photometric': ['gauss_s05', 'gauss_s10', 'gauss_s20', 'jpeg_q05', 'jpeg_q02', 'median_k05', 'median_k09', 'gamma_g080', 'gamma_g125', 'resize_f075', 'resize_f050'], 'geometric': ['rotate_d01', 'rotate_d03', 'rotkeep_d02', 'rotkeep_d05', 'cropkeep_f098', 'cropkeep_f095', 'cropkeep_f090', 'shift_f001', 'shift_f003', 'padedge_f002', 'padedge_f005'], 'overlay': ['text_s10']}

def output_path(path) -> Path:
    return Path(path).resolve()


def out_dir(step: str, smoke: bool=False) -> Path:
    d = output_path(RESULTS / ('smoke' if smoke else '') / step)
    d.mkdir(parents=True, exist_ok=True)
    return d

def atomic_json(path, data):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    for k in range(40):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if k == 39:
                raise
            time.sleep(0.05)

def save_csv(df: pd.DataFrame, path):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding='utf-8-sig')

def save_parquet(df: pd.DataFrame, path):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.parquet')
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)

def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()




def run_step(step, title, main, argv=None):
    ap = argparse.ArgumentParser(description=title)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--bootstrap', type=int, default=2000)
    ap.add_argument('--repeats', type=int, default=50)
    ap.add_argument('--scheme', default='phash')
    ap.add_argument('--part', default='all')
    args = ap.parse_args(argv)
    args.log = None
    target = RESULTS / ('smoke' if args.smoke else '') / step
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f'Choose an empty results directory before running {step}: {target}')
    main(args, out_dir(step, args.smoke))

_S = None

def runner():
    global _S
    if _S is None:
        sys.path.insert(0, str(MODULE_DIR))
        import predictor as S
        S.X.DEFAULT = REFERENCE
        import importlib
        for _name in ('conditional_predictor',):
            _mod = importlib.import_module('methods.' + _name)
            _mod.SOURCE = FISCH_SOURCE
            _mod.HASH_FILE = FISCH_HASH
        _S = S
    return _S

def modules():
    S = runner()
    from methods import fixed as MF
    import tasks as T
    import registry as REG
    return (S, MF, T, REG)

def load_reference(progress=None):
    S = runner()
    _, m, cfg = S.X.reference_inputs(S.X.DEFAULT)
    bundles, invalid = S.X.load_tasks(m)
    bundles = [b for b in bundles if b.scheme == 'dbp']
    fit, cal = S.R.split_train_cal([b for b in bundles if b.role == 'train'], 0.5)
    dev = [b for b in bundles if b.role == 'dev' and b.M in cfg['train_M']]
    test = [b for b in bundles if b.role == 'test']
    assert (len(fit), len(cal), len(dev)) == (324, 324, 568), (len(fit), len(cal), len(dev))
    assert sum((len(b.reg) for b in test)) == 1016
    return (S, m, cfg, fit, cal, dev, test)

def grid(w=None):
    S = runner()
    return S.SIM.grid_values(w or S.SIM.W_FIXED)

def succ(u: float, g: np.ndarray) -> float:
    if not np.isfinite(u):
        return float('inf')
    i = np.searchsorted(g, u, side='right')
    return float(g[i]) if i < len(g) else float('inf')

def prepare_test(test, fixed_dbp_tau, variant_names):
    for b in test:
        for d in b.reg.values():
            d['_sorted_normal'] = np.sort(np.asarray(d['Z_eval'], float))
            pv = np.asarray(d['pos_variant']).astype(str)
            zp = np.asarray(d['Z_pos'], float)
            d['_sorted_positive'] = {x: np.sort(zp[pv == x]) for x in variant_names}
            r = fast_eval(fixed_dbp_tau, d, 0.05, variant_names)
            d['_baseline_BA23'] = (r['TPR23'] + 1 - r['FAR']) / 2

def fast_eval(tau, d, alpha, variant_names):
    sentinel = not np.isfinite(tau)
    zz = d['_sorted_normal']
    far = 0.0 if sentinel else float((len(zz) - np.searchsorted(zz, tau, side='left')) / len(zz))
    r = {'tau': float(tau), 'sentinel': bool(sentinel), 'FAR': far, 'violation': bool(far > alpha + 1e-12), 'excess': far - alpha, 'n_eval': int(len(zz))}
    vals = []
    for name in variant_names:
        v = d['_sorted_positive'][name]
        t = (0.0 if sentinel else float((len(v) - np.searchsorted(v, tau, side='left')) / len(v))) if len(v) else np.nan
        r['TPR_' + name] = t
        vals.append(t)
    if not np.all(np.isfinite(vals)):
        raise ValueError('TPR is undefined for at least one edit type (no positive rows)')
    r['TPR23'] = float(np.mean(vals))
    r['BA23'] = (r['TPR23'] + 1 - r['FAR']) / 2
    return r

def row(b, kind, d, tau, n, alpha, method, variant_names, seed=-1, scheme='dbp', **extra):
    r = fast_eval(tau, d, alpha, variant_names)
    r.update(domain=b.domain, structure=b.structure, kind=kind, M=int(b.M), M_eff=int(d['M']), scheme=scheme, n=n, alpha=alpha, beta=round(1 - alpha, 2), method=method, training_seed=seed, seed=int(b.seed), boundary_case=False, baseline_BA23=d['_baseline_BA23'])
    r.update(extra)
    return r

def condition_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(K, dropna=False):
        z = dict(zip(K, key))
        z.update(n_tasks=len(g), V=float(g.violation.mean()), FAR=float(g.FAR.mean()), TPR23=float(g.TPR23.mean()), BA23=float(g.BA23.mean()), baseline_BA23=float(g.baseline_BA23.mean()), sentinel_rate=float(g.sentinel.mean()))
        z['meets_far'] = z['V'] <= DELTA + 1e-12
        z['meets_perf'] = z['BA23'] >= z['baseline_BA23'] - 1e-12
        z['meets'] = z['meets_far'] and z['meets_perf']
        z['no_detection'] = z['TPR23'] <= 0
        rows.append(z)
    return pd.DataFrame(rows)

def method_summary(cond: pd.DataFrame, by=('method', 'n', 'alpha')) -> pd.DataFrame:
    by = list(by)
    ps = cond.groupby(by + ['training_seed']).agg(conditions=('meets', 'size'), passing=('meets', 'sum'), far_ok=('meets_far', 'sum'), FAR=('FAR', 'mean'), V=('V', 'mean'), TPR23=('TPR23', 'mean'), BA23=('BA23', 'mean')).reset_index()
    out = ps.groupby(by).agg(seeds=('training_seed', 'nunique'), conditions=('conditions', 'first'), passing=('passing', 'mean'), passing_sd=('passing', 'std'), far_ok=('far_ok', 'mean'), FAR=('FAR', 'mean'), V=('V', 'mean'), TPR23=('TPR23', 'mean'), BA23=('BA23', 'mean')).reset_index()
    return out

def hash_convention_rows(smoke=False):
    p = stored(RESULTS / ('smoke' if smoke else '') / 'hash_thresholds' / 'hash_convention_rows.parquet')
    return pd.read_parquet(p) if p.exists() else None

def arm_path(n, beta, mode=None, seed=None, cal=None, baseline=False):
    tag = f'b{int(round(beta * 100)):02d}'
    if baseline:
        return stored(ARMS / f'baseline_n{n}_{tag}.parquet')
    return stored(ARMS / f'n{n}_{tag}_{mode}_s{seed}_{cal}.parquet')

def read_arms(n, alpha, modes=('base_set',), cals=('np_global', 'fisch_global'), seeds=SEEDS, baselines=True):
    beta = round(1 - alpha, 2)
    frames = []
    for mode in modes:
        for c in cals:
            for s in seeds:
                p = arm_path(n, beta, mode, s, c)
                if not p.exists():
                    raise FileNotFoundError(p)
                frames.append(pd.read_parquet(p))
    if baselines:
        p = arm_path(n, beta, baseline=True)
        if not p.exists():
            raise FileNotFoundError(p)
        frames.append(pd.read_parquet(p))
    return pd.concat(frames, ignore_index=True)

def encode(S, d, alpha, mode, n):
    z = np.asarray(d['Z_sup'], float)[:n]
    if len(z) != n or not np.isfinite(z).all():
        raise ValueError(f'support set needs {n} finite scores')
    c = np.log10(d['M'])
    if mode.endswith('_set'):
        return np.column_stack((z, np.full(n, c)))
    return np.r_[S.F.SUMMARY.features(z, n, d['M'], alpha)[:7], c]

def load_model(S, n, mode, alpha, seed):
    import torch
    path = MODELS / f'n{n}' / f'{mode}__{VAR[mode]}__a{alpha}__s{seed}'
    md_p, pt = (path / 'metadata.json', path / 'model.pt')
    if not (md_p.exists() and pt.exists()):
        raise FileNotFoundError(f'trained model missing: {path}')
    md = json.loads(md_p.read_text(encoding='utf-8-sig'))
    if md['mode'] != mode or md['alpha'] != alpha or md['training_seed'] != seed:
        raise ValueError(f'stored model metadata does not match mode={mode}, alpha={alpha}, seed={seed}: {path}')
    torch.manual_seed(seed)
    obj = S.Predictor(mode, md['mu'], md['sd'], md['pool'])
    obj.load_state_dict(torch.load(pt, map_location='cpu', weights_only=True))
    obj.eval()
    return (obj, md, path)

def predict(S, obj, d, alpha, mode, n):
    import torch
    with torch.no_grad():
        x = torch.tensor(encode(S, d, alpha, mode, n)[None], dtype=torch.float32)
        return float(obj(x)[0])

def target_quantile(S, Z_eval, alpha):
    return S.FC.empirical_quantile(Z_eval, S.FC.beta_fraction(alpha))

def residual_lambda(S, rows, alpha):
    return S.residual_cal(rows, alpha, 'np')
RAW = (ROOT / Path(PATHS_CFG.get('raw_root') or 'data/raw')).resolve()
SOLANA_IMG = (ROOT / Path(PATHS_CFG.get('solana_img_root') or 'data/solana')).resolve()
ROOTS = {'solana': SOLANA_IMG, 'nft': RAW / 'nft_classifier', 'disc21_bg': RAW / 'disc21' / 'references' / 'images' / 'references'}
DISC21 = RAW / 'disc21'
P02 = ROOT / 'data' / 'hashes'
WEIGHTS = ROOT / 'weights' / 'sscd_disc_mixup.torchscript.pt'
WEIGHT_URLS = ['https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt', 'https://huggingface.co/m3/sscd-copy-detection/resolve/main/sscd_disc_mixup.torchscript.pt']
SSCD_SIZE = 288
SSCD_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
SSCD_STD = np.array([0.229, 0.224, 0.225], np.float32)
EMB = ROOT / 'data' / 'sscd_emb'
_DEV = None

def runtime_modules():
    rt = MODULE_DIR / 'core'
    for p in (str(rt), str(rt)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import dbphash as D, hashcache as HC, transforms as TR
    return (D, HC, TR)

def cont_succ(u: float) -> float:
    if not np.isfinite(u):
        return float('inf')
    return float(np.nextafter(float(u), np.inf))

def sscd_device():
    import torch
    want = os.environ.get('SSCD_DEVICE', 'auto').lower()
    if want == 'cpu':
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    if want == 'cuda':
        raise RuntimeError('SSCD_DEVICE=cuda but torch.cuda.is_available() is False (CPU-only torch or no driver)')
    return torch.device('cpu')

def sscd_load():
    global _DEV
    import torch
    if not WEIGHTS.exists():
        raise FileNotFoundError(f'SSCD weights missing: run the sscd_download step first ({WEIGHTS})')
    _DEV = sscd_device()
    if _DEV.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.set_num_threads(2)
    else:
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    m = torch.jit.load(str(WEIGHTS), map_location=_DEV)
    m.eval()
    return m

def sscd_runtime_info():
    import torch
    d = _DEV or sscd_device()
    return {'device': str(d), 'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0) if d.type == 'cuda' else None, 'tf32': False if d.type == 'cuda' else None, 'precision': 'float32'}

def sscd_tensor(img_bgr):
    import torch, cv2
    from PIL import Image
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    w, h = pil.size
    if w <= h:
        nw, nh = (SSCD_SIZE, max(1, int(SSCD_SIZE * h / w)))
    else:
        nw, nh = (max(1, int(SSCD_SIZE * w / h)), SSCD_SIZE)
    pil = pil.resize((nw, nh), Image.BILINEAR)
    x = (np.asarray(pil, np.float32) / 255.0 - SSCD_MEAN) / SSCD_STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None]

def sscd_embed_tensors(model, tensors, max_batch=32) -> np.ndarray:
    import torch
    dev = _DEV or torch.device('cpu')
    out = np.empty((len(tensors), 0), np.float32) if not tensors else None
    groups = {}
    for i, t in enumerate(tensors):
        groups.setdefault(tuple(t.shape), []).append(i)
    res = [None] * len(tensors)
    with torch.no_grad():
        for idx in groups.values():
            for k in range(0, len(idx), max_batch):
                part = idx[k:k + max_batch]
                x = torch.cat([tensors[i] for i in part], 0).to(dev, non_blocking=False)
                v = model(x).float().cpu().numpy()
                for j, i in enumerate(part):
                    res[i] = v[j]
    if out is not None:
        return out
    V = np.stack(res).astype(np.float32)
    n = np.linalg.norm(V, axis=1, keepdims=True)
    return np.where(n > 0, V / np.where(n > 0, n, 1), V).astype(np.float32)

def sscd_embed(model, img_bgr) -> np.ndarray:
    return sscd_embed_tensors(model, [sscd_tensor(img_bgr)])[0]

def load_bgr_bytes(buf: bytes):
    import cv2
    im = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    if im.dtype != np.uint8:
        im = (im.astype(np.float64) / (np.iinfo(im.dtype).max / 255.0) + 0.5).astype(np.uint8)
    if im.ndim == 2:
        return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    if im.shape[2] == 4:
        a = im[:, :, 3:4].astype(np.float32) / 255.0
        return np.clip(im[:, :, :3].astype(np.float32) * a + 255.0 * (1.0 - a) + 0.5, 0, 255).astype(np.uint8)
    return im if im.shape[2] == 3 else None

class EmbStore:

    def __init__(self, folder: Path, index: pd.DataFrame | None=None, dim: int | None=None):
        self.folder = output_path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        ip = self.folder / 'index.parquet'
        if index is not None and (not ip.exists()):
            index = index.reset_index(drop=True)
            save_parquet(index, ip)
            np.lib.format.open_memmap(self.folder / 'emb.npy', mode='w+', dtype=np.float32, shape=(len(index), dim))
            np.save(self.folder / 'done.npy', np.zeros(len(index), bool))
        self.index = pd.read_parquet(ip)
        if index is not None and len(index) != len(self.index):
            raise RuntimeError(f'index changed in {self.folder}')
        self.emb = np.load(self.folder / 'emb.npy', mmap_mode='r+')
        self.done = np.load(self.folder / 'done.npy')

    def save_done(self):
        self.emb.flush()
        tmp = self.folder / 'done.tmp.npy'
        np.save(tmp, self.done)
        os.replace(tmp, self.folder / 'done.npy')

    @staticmethod
    def open_read(folder: Path):
        idx = pd.read_parquet(folder / 'index.parquet')
        done = np.load(folder / 'done.npy')
        if not done.all():
            raise RuntimeError(f'embeddings incomplete in {folder}: {int((~done).sum())} missing')
        return (idx, np.load(folder / 'emb.npy', mmap_mode='r'))

def max_cos(Q: np.ndarray, R: np.ndarray, chunk: int=2048) -> np.ndarray:
    out = np.empty(len(Q), np.float64)
    if len(R) == 0:
        out[:] = -np.inf
        return out
    Rt = np.ascontiguousarray(R, dtype=np.float32).T
    for i in range(0, len(Q), chunk):
        out[i:i + chunk] = (np.asarray(Q[i:i + chunk], np.float32) @ Rt).max(axis=1)
    return out

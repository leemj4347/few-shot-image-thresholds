# Prepare experiment inputs and author source
from pathlib import Path
import argparse, hashlib, json, shutil
ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stage(source, records, destination):
    source = source.resolve()
    destination = destination.resolve()
    for record in records:
        relative = Path(record.get("path", record.get("file")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid relative input path")
        src = (source / relative).resolve()
        dst = (destination / relative).resolve()
        if source not in src.parents or destination not in dst.parents:
            raise ValueError("Input path escapes root")
        if not src.is_file():
            raise FileNotFoundError(src)
        if digest(src).lower() != record["sha256"].lower():
            raise ValueError("Input SHA256 mismatch: " + str(relative))
    for record in records:
        relative = Path(record.get("path", record.get("file")))
        src = source / relative
        dst = destination / relative
        if src.resolve() == dst.resolve():
            continue
        if dst.exists():
            if digest(dst).lower() != record["sha256"].lower():
                raise FileExistsError("Different file exists: " + str(dst))
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if digest(dst).lower() != record["sha256"].lower():
            raise RuntimeError("Copy mismatch: " + str(relative))
    print("Prepared", len(records), "input files")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path)
    p.add_argument("--fisch-source", type=Path)
    args = p.parse_args()
    if not args.source and (not args.fisch_source):
        p.error("Specify --source or --fisch-source")
    if args.source:
        stage(args.source, json.loads((ROOT / "configs/inputs.json").read_text(encoding="utf-8")), ROOT)
    if args.fisch_source:
        dep = json.loads((ROOT / "configs/dependencies.json").read_text(encoding="utf-8"))["fisch"]
        stage(args.fisch_source, dep["files"], ROOT / "external/fisch")
        (ROOT / "external/fisch/source_hashes.json").write_text(json.dumps(dep["files"], indent=2), encoding="utf-8")
if __name__ == "__main__":
    main()

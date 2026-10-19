"""Extract frozen timm features and evaluate PS and linear-probe CG.

Example:
    python corr_ps_cg.py --dataset PACS --data-root datasets/PACS --output-dir results
    python corr_ps_cg.py --dataset PACS --stage metrics --output-dir results
"""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression

DATASETS = {"PACS": (4, 7), "OfficeHome": (4, 65), "NICO": (6, 60)}
DEFAULT_MODELS = ["vit_base_patch16_224.augreg_in1k", "vit_base_patch16_224.augreg_in21k",
                  "vit_base_patch16_224.dino", "vit_base_patch16_clip_224.openai"]


def feature_path(folder, model):
    return folder / (model.replace(":", "__").replace("/", "_") + ".npz")


def extract_features(args, name, destination):
    """Cache raw pooled features, labels, image paths, and model identity."""
    import timm
    import torch
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    root = args.data_root.resolve()
    root = root / "images" if (root / "images").is_dir() else root
    domains = sorted([p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")])
    nd, nc = DATASETS[args.dataset]
    if len(domains) != nd:
        raise ValueError(f"{args.dataset} expects {nd} domains under {root}; found {domains}")
    pretrained = not name.startswith("random:")
    architecture = name.removeprefix("random:")
    device = torch.device(args.device)
    model = timm.create_model(architecture, pretrained=pretrained, num_classes=0)
    config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**config, is_training=False)
    model.requires_grad_(False).to(device).eval()
    features, class_labels, domain_labels, paths = [], [], [], []
    classes = None
    for domain_id, domain in enumerate(domains):
        dataset = ImageFolder(root / domain, transform=transform)
        if classes is None:
            classes = dataset.classes
        if dataset.classes != classes or len(classes) != nc:
            raise ValueError(f"Every domain must have the same {nc} class folders")
        paths.extend(Path(path).relative_to(root).as_posix() for path, _ in dataset.samples)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=device.type == "cuda")
        with torch.inference_mode():
            for images, labels in loader:
                batch = model(images.to(device, non_blocking=True))
                if batch.ndim != 2:
                    raise ValueError(f"Expected pooled [N, D] features; got {batch.shape}")
                features.append(batch.cpu().numpy())
                class_labels.append(labels.numpy())
                domain_labels.append(np.full(len(labels), domain_id, dtype=np.int64))
        print(f"  {domain}: {len(dataset)} images", flush=True)
    metadata = dict(model=name, dataset=args.dataset, pretrained=pretrained, seed=args.seed,
                    data_root=str(args.data_root.resolve()), data_config=config,
                    versions=dict(torch=torch.__version__, timm=timm.__version__, numpy=np.__version__))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.with_suffix(".tmp").open("wb") as stream:
        np.savez(stream, features=np.concatenate(features), class_labels=np.concatenate(class_labels),
                 domain_labels=np.concatenate(domain_labels), image_paths=np.asarray(paths),
                 classes=np.asarray(classes), domains=np.asarray(domains),
                 metadata=np.asarray(json.dumps(metadata, default=str)))
    destination.with_suffix(".tmp").replace(destination)


def load_features(path, dataset, model):
    with np.load(path, allow_pickle=False) as data:
        x, yc, yd = data["features"], data["class_labels"], data["domain_labels"]
        metadata = json.loads(data["metadata"].item())
        identity = (data["image_paths"].tolist(), yc.tolist(), yd.tolist(),
                    data["classes"].tolist(), data["domains"].tolist())
    nd, nc = DATASETS[dataset]
    if metadata["model"] != model or metadata["dataset"] != dataset:
        raise ValueError(f"Cache identity mismatch: {path}")
    if x.ndim != 2 or yc.shape != (len(x),) or yd.shape != (len(x),) or not np.isfinite(x).all():
        raise ValueError(f"Invalid features or labels: {path}")
    if set(yc.tolist()) != set(range(nc)) or set(yd.tolist()) != set(range(nd)):
        raise ValueError(f"Expected contiguous labels for {nd} domains and {nc} classes")
    if np.any(np.bincount(yd * nc + yc, minlength=nd * nc) == 0):
        raise ValueError("The survey requires samples for every domain-class combination")
    return x, yc, yd, metadata, identity


def make_plan(nd, nc, ps_samples=100, cg_splits=10, seed=42):
    """Sample one plan and reuse it for every backbone."""
    rng = random.Random(seed)
    plan = dict(ps=[[rng.sample(range(nd), 2), rng.sample(range(nc), 2)] for _ in range(ps_samples)],
                test_domains=[])
    for _ in range(cg_splits):
        held = [rng.randrange(nd) for _ in range(nc)]
        while len(set(held)) == 1:
            held = [rng.randrange(nd) for _ in range(nc)]
        plan["test_domains"].append(held)
    return plan


def calculate_ps(x, yc, yd, plan):
    centroids = {(int(d), int(c)): x[(yd == d) & (yc == c)].mean(0)
                 for d in np.unique(yd) for c in np.unique(yc)}

    def cosine(u, v):
        value = np.dot(u, v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-12)
        return float(np.clip(value, -1, 1))

    pc, pd = [], []
    for (d1, d2), (c1, c2) in plan["ps"]:
        a, b, c, d = [centroids[key] for key in ((d1, c1), (d1, c2), (d2, c1), (d2, c2))]
        pc.append(cosine(a-b, c-d))
        pd.append(cosine(a-c, b-d))
    return pc, pd


def calculate_cg(x, yc, yd, plan, max_iter=10000):
    """Fit separate multinomial linear probes for class and domain prediction."""
    def accuracy(labels, test):
        probe = LogisticRegression(C=1.0, solver="lbfgs", max_iter=max_iter, tol=1e-4)
        probe.fit(x[~test], labels[~test])
        return float(np.mean(probe.predict(x[test]) == labels[test]))

    class_scores, domain_scores = [], []
    for split in plan["test_domains"]:
        test = yd == np.asarray(split)[yc]
        class_scores.append(accuracy(yc, test))
        domain_scores.append(accuracy(yd, test))
    return class_scores, domain_scores


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def correlations(rows):
    """Undefined correlations (one model or constant inputs) are left blank."""
    output = []
    for setting in ("all", "pretrained"):
        selected = [r for r in rows if setting == "all" or not r["Model"].startswith("random:")]
        for concept in ("class", "domain"):
            x = [r[f"PS-{concept}"] for r in selected]
            y = [r[f"CG-{concept}"] for r in selected]
            valid = len(x) >= 2 and np.ptp(x) > 0 and np.ptp(y) > 0
            output.append(dict(Models=setting, Concept=concept, n=len(x),
                               Pearson=float(pearsonr(x, y).statistic) if valid else "",
                               Spearman=float(spearmanr(x, y).statistic) if valid else ""))
    return output


def run(args):
    root = args.output_dir / args.dataset
    folder = root / "features"
    folder.mkdir(parents=True, exist_ok=True)
    models = args.models
    if args.model_list:
        models = [line.split("#", 1)[0].strip() for line in args.model_list.read_text(encoding="utf-8-sig").splitlines()]
        models = [name for name in models if name]
    if models is None:
        if args.stage == "metrics":
            models = []
            for path in sorted(folder.glob("*.npz")):
                with np.load(path, allow_pickle=False) as data:
                    models.append(json.loads(data["metadata"].item())["model"])
        else:
            models = DEFAULT_MODELS
    if not models or len(models) != len(set(models)):
        raise ValueError("Select a nonempty, unique model list, or provide existing caches")
    paths = [feature_path(folder, name) for name in models]
    if len(paths) != len(set(paths)):
        raise ValueError("Model names map to colliding cache filenames")
    plan = make_plan(*DATASETS[args.dataset], args.ps_samples, args.cg_splits, args.seed)
    report = dict(dataset=args.dataset, seed=args.seed,
                  ps_samples=args.ps_samples, cg_splits=args.cg_splits, max_iter=args.max_iter,
                  splits=plan, results={})
    reference, rows = None, []
    for number, (name, path) in enumerate(zip(models, paths), 1):
        print(f"[{number}/{len(models)}] {name}", flush=True)
        if args.stage != "metrics" and (args.overwrite or not path.exists()):
            extract_features(args, name, path)
        x, yc, yd, metadata, identity = load_features(path, args.dataset, name)
        if args.stage != "metrics" and (metadata["seed"] != args.seed or metadata["data_root"] != str(args.data_root.resolve())):
            raise ValueError("Cache seed/root differs; use --overwrite or the original extraction settings")
        if reference is not None and identity != reference:
            raise ValueError("Images or label ordering differ between models")
        reference = identity
        if args.stage == "extract":
            continue
        pc, pd = calculate_ps(x, yc, yd, plan)
        cc, cd = calculate_cg(x, yc, yd, plan, args.max_iter)
        row = dict(Model=name, **{"PS-class": float(np.mean(pc)), "CG-class": float(np.mean(cc)),
                                 "PS-domain": float(np.mean(pd)), "CG-domain": float(np.mean(cd))})
        rows.append(row)
        report["results"][name] = dict(PS_class=pc, PS_domain=pd, CG_class=cc, CG_domain=cd, extraction=metadata)
        write_csv(root / "metrics.csv", rows)
        write_csv(root / "correlations.csv", correlations(rows))
        (root / "details.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(row, flush=True)
    print(f"Saved outputs under {root}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--data-root", type=Path, help="Required for all/extract stages.")
    p.add_argument("--output-dir", type=Path, default=Path("corr_results"))
    p.add_argument("--stage", choices=("all", "extract", "metrics"), default="all")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--models", nargs="+", help="Exact timm names; random:ARCH selects an untrained control.")
    group.add_argument("--model-list", type=Path, help="UTF-8 text, one model per line; # starts a comment.")
    p.add_argument("--device", default="cpu", help="Feature extraction device, e.g. cpu or cuda:0.")
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--workers", type=int, default=0, help="Number of data loading workers.")
    p.add_argument("--ps-samples", type=int, default=100)
    p.add_argument("--cg-splits", type=int, default=10,
                   help="Random CG splits per model: fit one class and one domain probe per split (default: 10).")
    p.add_argument("--max-iter", type=int, default=10000, help="Logistic regression iteration limit.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true", help="Re-extract existing feature caches.")
    args = p.parse_args()
    if args.stage != "metrics" and args.data_root is None:
        p.error("--data-root is required for all/extract")
    if min(args.batch_size, args.ps_samples, args.cg_splits, args.max_iter) < 1 or args.workers < 0:
        p.error("Counts must be positive and workers nonnegative")
    return args


if __name__ == "__main__":
    run(parse_args())

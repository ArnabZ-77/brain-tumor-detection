"""Internal vs EXTERNAL evaluation (Section 8). Run each checkpoint once on the external set.

  python evaluate.py --ckpt runs/proposed --external path/to/Br35H
  python evaluate.py --ckpt runs/resnet50  --external path/to/Br35H --mc 1 --no-tta   # plain baseline
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from calibration import accuracy_coverage, ece, triage
from data import CLASSES, NOTUMOR, MRIDataset, load_splits, scan_external
from model import BrainTumorNet, predict_loader


def collapse_binary(probs):
    """4-class probs -> (P(tumor), predicted tumor?)."""
    p_tumor = 1.0 - probs[:, NOTUMOR]
    return p_tumor, (p_tumor > 0.5).astype(int)


def summarize(r, binary, tau):
    """r: predict_loader output. Returns metrics dict + per-sample `correct`."""
    labels, probs = r["labels"], r["probs"]
    if binary:
        _, pred = collapse_binary(probs)
        correct = (pred == labels)
        recall = {"tumor": float(correct[labels == 1].mean()), "notumor": float(correct[labels == 0].mean())}
        calib = None  # top-label ECE is defined on the 4-class output only
    else:
        pred = probs.argmax(1)
        correct = pred == labels
        recall = {c: float(correct[labels == i].mean()) for i, c in enumerate(CLASSES) if (labels == i).any()}
        calib = ece(probs, labels)
    return {"accuracy": float(correct.mean()), "per_class_recall": recall, "ece": calib,
            "triage": triage(r["entropy"], correct, tau)}, correct


def loader_for(items, bs, workers):
    return DataLoader(MRIDataset(items), batch_size=bs, num_workers=workers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="run dir containing model.pt and split.json")
    ap.add_argument("--external", required=True)
    ap.add_argument("--mc", type=int, default=25, help="MC-dropout passes (1 = deterministic)")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--no-calibration", action="store_true", help="use T=1")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    run = Path(a.ckpt)
    ck = torch.load(run / "model.pt", map_location="cpu")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = BrainTumorNet(ck["arch"], len(CLASSES), pretrained=False)
    model.load_state_dict(ck["state"]); model.to(dev).eval()
    T = 1.0 if a.no_calibration else ck["temperature"]
    kw = dict(temperature=T, n_mc=a.mc, tta=not a.no_tta)

    # Internal (held-out split of the training dataset)
    splits = load_splits(run / "split.json")
    t0 = time.time()
    internal = predict_loader(model, loader_for(splits["test"], a.bs, a.workers), dev, **kw)
    ms_per_img = (time.time() - t0) / len(splits["test"]) * 1000
    int_m, int_ok = summarize(internal, False, ck["tau"])
    # Same internal data collapsed to tumor / no-tumor, so it is comparable with Br35H.
    _, ip = collapse_binary(internal["probs"])
    int_bin = float((ip == (internal["labels"] != NOTUMOR).astype(int)).mean())

    # External (never seen, never tuned on)
    ext_items, binary = scan_external(a.external)
    ext = predict_loader(model, loader_for(ext_items, a.bs, a.workers), dev, **kw)
    ext_m, ext_ok = summarize(ext, binary, ck["tau"])

    ref_internal = int_bin if binary else int_m["accuracy"]
    res = {
        "arch": ck["arch"], "temperature": T, "mc_passes": a.mc,
        "internal": int_m, "internal_binary_accuracy": int_bin, "external": ext_m,
        "external_is_binary": binary,
        "generalization_gap_delta": ref_internal - ext_m["accuracy"],
        "params_millions": sum(p.numel() for p in model.parameters()) / 1e6,
        "ms_per_image_incl_mc": ms_per_img,
    }
    (run / "eval.json").write_text(json.dumps(res, indent=2))

    with open(run / "accuracy_coverage_external.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["coverage", "accuracy"])
        w.writerows(accuracy_coverage(ext["entropy"], ext_ok))

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for name, r, ok in (("internal", internal, int_ok), ("external", ext, ext_ok)):
            c = np.array(accuracy_coverage(r["entropy"], ok))
            plt.plot(c[:, 0], c[:, 1], label=name)
        plt.xlabel("coverage (fraction auto-decided)"); plt.ylabel("accuracy on covered cases")
        plt.legend(); plt.grid(alpha=.3); plt.savefig(run / "accuracy_coverage.png", dpi=150)
    except ImportError:
        pass

    print(json.dumps(res, indent=2))
    print(f"\nInternal acc {ref_internal:.4f} | External acc {ext_m['accuracy']:.4f} | "
          f"Delta {res['generalization_gap_delta']:.4f}")


if __name__ == "__main__":
    main()

"""Train on ONE dataset (Kaggle/Masoud). The external set is never touched here.

  python train.py --data path/to/kaggle --out runs/proposed
  python train.py --data path/to/kaggle --out runs/resnet50 --arch resnet50 --no-acq   # baseline
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from calibration import ece, fit_temperature, threshold_for_coverage
from data import CLASSES, MRIDataset, make_splits, save_splits, scan_four_class
from model import BrainTumorNet, ensure_cpu_state, predict_loader


class EMA:
    """Exponential moving average of weights."""

    def __init__(self, model, decay=0.999):
        self.module, self.decay = copy.deepcopy(model).eval(), decay

    @torch.no_grad()
    def update(self, model):
        for e, m in zip(self.module.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(self.decay).add_(m.detach(), alpha=1 - self.decay)
            else:
                e.copy_(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="runs/proposed")
    ap.add_argument("--arch", default="mobilevitv2_100")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--coverage", type=float, default=0.90, help="target auto-coverage for tau")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-acq", action="store_true", help="disable acquisition-aware aug (baseline)")
    ap.add_argument("--no-pretrained", action="store_true")
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    splits = make_splits(scan_four_class(a.data), a.seed)
    save_splits(splits, out / "split.json")
    print({k: len(v) for k, v in splits.items()})

    mk = lambda items, train: DataLoader(
        MRIDataset(items, train=train, acquisition=not a.no_acq), batch_size=a.bs,
        shuffle=train, num_workers=a.workers, pin_memory=dev == "cuda",
        persistent_workers=a.workers > 0)
    tr, va = mk(splits["train"], True), mk(splits["val"], False)

    model = BrainTumorNet(a.arch, len(CLASSES), pretrained=not a.no_pretrained).to(dev)
    ema = EMA(model)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * len(tr))
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    amp = dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best, best_state, bad = float("inf"), None, 0
    for ep in range(a.epochs):
        model.train()
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(dev, enabled=amp):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            sched.step(); ema.update(model)

        ema.module.to(dev)
        v = predict_loader(ema.module, va, dev, n_mc=1, tta=False)
        vl = float(crit(torch.from_numpy(v["logits"]), torch.from_numpy(v["labels"]).long()))
        va_acc = float((v["logits"].argmax(1) == v["labels"]).mean())
        print(f"epoch {ep + 1:02d}  train_loss {loss.item():.4f}  val_loss {vl:.4f}  val_acc {va_acc:.4f}")
        if vl < best:
            best, bad = vl, 0
            best_state = ensure_cpu_state(ema.module.state_dict())
        else:
            bad += 1
            if bad >= a.patience:
                print("early stopping"); break

    # Calibrate + pick triage threshold on VALIDATION only.
    ema.module.load_state_dict(best_state); ema.module.to(dev)
    v = predict_loader(ema.module, va, dev, n_mc=1, tta=False)
    T = fit_temperature(v["logits"], v["labels"])
    v = predict_loader(ema.module, va, dev, temperature=T, n_mc=25, tta=True)
    tau = threshold_for_coverage(v["entropy"], a.coverage)
    print(f"temperature {T:.3f}  tau@{a.coverage:.0%} coverage {tau:.4f}  val ECE {ece(v['probs'], v['labels']):.4f}")

    torch.save({"arch": a.arch, "state": best_state, "temperature": T, "tau": tau,
                "coverage": a.coverage, "classes": CLASSES}, out / "model.pt")
    (out / "train_summary.json").write_text(json.dumps(
        {"best_val_loss": best, "temperature": T, "tau": tau, "args": vars(a)}, indent=2))


if __name__ == "__main__":
    main()

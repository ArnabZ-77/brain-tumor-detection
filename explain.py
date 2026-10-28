"""Grad-CAM++ and quantitative explanation scoring (Step 7).

  python explain.py --ckpt runs/proposed --image scan.jpg [--mask tumor_mask.png]

Needs a backbone whose forward_features returns a 4D map (MobileViT-V2, ResNet).
Attention Rollout from the report is not included: timm's MobileViT attention is
not exposed per-patch; for a ViT/Swin backbone add it on top of this file.
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from data import CLASSES
from model import BrainTumorNet


def gradcam_pp(model, x, class_idx=None):
    """x: (B,3,H,W). Returns cam (B,H,W) in [0,1] and the class index used."""
    model.eval()
    x = x.requires_grad_(True)
    fmap = model.backbone.forward_features(x)
    fmap.retain_grad()
    logits = model.head(fmap.mean((2, 3)))
    idx = logits.argmax(1) if class_idx is None else torch.full((len(x),), class_idx, device=x.device)
    score = logits.gather(1, idx[:, None]).sum()
    model.zero_grad(); score.backward()

    g, A = fmap.grad, fmap.detach()
    g2, g3 = g ** 2, g ** 3
    denom = 2 * g2 + A.sum((2, 3), keepdim=True) * g3
    alpha = g2 / torch.where(denom != 0, denom, torch.ones_like(denom))
    w = (alpha * F.relu(g)).sum((2, 3), keepdim=True)
    cam = F.relu((w * A).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    mn = cam.amin((1, 2), keepdim=True); mx = cam.amax((1, 2), keepdim=True)
    return ((cam - mn) / (mx - mn + 1e-8)).detach(), idx


def explanation_scores(cam, mask, thr=0.5):
    """IoU and Dice between thresholded heatmap and ground-truth tumor mask (both HxW)."""
    pred = np.asarray(cam) >= thr
    gt = np.asarray(mask) > 0
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return {"iou": inter / union if union else 0.0,
            "dice": 2 * inter / (pred.sum() + gt.sum()) if (pred.sum() + gt.sum()) else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--mask")
    ap.add_argument("--out", default="gradcam.png")
    a = ap.parse_args()

    ck = torch.load(f"{a.ckpt}/model.pt", map_location="cpu")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = BrainTumorNet(ck["arch"], len(CLASSES), pretrained=False)
    model.load_state_dict(ck["state"]); model.to(dev)

    img = Image.open(a.image).convert("L").resize((224, 224), Image.BILINEAR)
    g = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
    x = ((g - g.mean()) / (g.std() + 1e-6))[None, None].repeat(1, 3, 1, 1).to(dev)

    cam, idx = gradcam_pp(model, x)
    print("predicted:", CLASSES[int(idx[0])])
    cam = cam[0].cpu().numpy()
    if a.mask:
        m = np.asarray(Image.open(a.mask).convert("L").resize((224, 224), Image.NEAREST))
        print("explanation scores:", explanation_scores(cam, m))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.imshow(np.asarray(img), cmap="gray"); plt.imshow(cam, cmap="jet", alpha=0.4)
    plt.axis("off"); plt.savefig(a.out, bbox_inches="tight", dpi=150)


if __name__ == "__main__":
    main()

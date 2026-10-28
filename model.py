"""Hybrid CNN-Transformer classifier with Monte-Carlo-Dropout uncertainty (Steps 4-5)."""
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class BrainTumorNet(nn.Module):
    """ImageNet-pretrained timm backbone (default MobileViT-V2) + dropout head.

    Any timm arch works, so the same class builds the baselines:
    'resnet50', 'vit_base_patch16_224', 'mobilevitv2_100', 'swin_tiny_patch4_window7_224'.
    """

    def __init__(self, arch="mobilevitv2_100", num_classes=4, pretrained=True, drop=0.3):
        super().__init__()
        self.arch = arch
        self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0)
        d = self.backbone.num_features
        self.head = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Dropout(drop),
                                  nn.Linear(256, num_classes))

    def features(self, x):
        return self.backbone(x)

    def forward(self, x):
        return self.head(self.features(x))


def _set_head_dropout(model, active):
    for m in model.head.modules():
        if isinstance(m, nn.Dropout):
            m.train(active)


def _entropy(p):
    return -(p * torch.log(p.clamp_min(1e-12))).sum(-1)


@torch.no_grad()
def predict_loader(model, loader, device, temperature=1.0, n_mc=25, tta=True):
    """Runs the backbone once per batch (optionally with an h-flip TTA average) and
    the dropout head `n_mc` times.

    Returns numpy dict:
      logits  deterministic (dropout off) logits, un-scaled  -> fit temperature on these
      probs   MC-mean softmax(logits / T)                      -> final prediction
      entropy predictive entropy of probs                      -> uncertainty score
      mi      mutual information (epistemic part)
      labels
    Backbone BN stays in eval mode; only the head dropout is stochastic.
    """
    model.eval()
    out = {k: [] for k in ("logits", "probs", "entropy", "mi", "labels")}
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        f = model.features(x)
        if tta:
            f = (f + model.features(torch.flip(x, dims=[-1]))) / 2
        det = model.head(f)
        if n_mc > 1:
            _set_head_dropout(model, True)
            ps = torch.stack([F.softmax(model.head(f) / temperature, -1) for _ in range(n_mc)])
            _set_head_dropout(model, False)
            p = ps.mean(0)
            mi = _entropy(p) - _entropy(ps).mean(0)
        else:
            p = F.softmax(det / temperature, -1)
            mi = torch.zeros(len(p), device=p.device)
        out["logits"].append(det.cpu()); out["probs"].append(p.cpu())
        out["entropy"].append(_entropy(p).cpu()); out["mi"].append(mi.cpu())
        out["labels"].append(y)
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def ensure_cpu_state(state):
    return {k: v.detach().cpu() for k, v in state.items()}

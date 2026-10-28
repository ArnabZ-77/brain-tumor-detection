"""Temperature scaling (Step 6), ECE, and accuracy-vs-coverage triage (Step 8)."""
import numpy as np
import torch
import torch.nn.functional as F


def fit_temperature(logits, labels):
    """Single scalar T minimising NLL of softmax(logits / T) on the validation set."""
    logits = torch.as_tensor(logits, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.long)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp())


def ece(probs, labels, n_bins=15):
    """Expected Calibration Error (top-label)."""
    probs, labels = np.asarray(probs), np.asarray(labels)
    conf, pred = probs.max(1), probs.argmax(1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def accuracy_coverage(uncertainty, correct, n_points=50):
    """Keep the most certain fraction of cases; report accuracy on that fraction."""
    uncertainty, correct = np.asarray(uncertainty), np.asarray(correct, dtype=float)
    order = np.argsort(uncertainty, kind="stable")
    c = correct[order]
    cum = np.cumsum(c) / np.arange(1, len(c) + 1)
    ks = np.unique(np.linspace(max(1, len(c) // n_points), len(c), n_points).astype(int))
    return [(k / len(c), float(cum[k - 1])) for k in ks]


def threshold_for_coverage(uncertainty_val, coverage=0.90):
    """tau such that `coverage` of validation cases fall below it. Fit on VAL only."""
    return float(np.quantile(uncertainty_val, coverage))


def triage(uncertainty, correct, tau):
    """Cases with uncertainty >= tau are referred to a radiologist."""
    uncertainty, correct = np.asarray(uncertainty), np.asarray(correct, dtype=float)
    auto = uncertainty < tau
    return {
        "coverage": float(auto.mean()),
        "auto_accuracy": float(correct[auto].mean()) if auto.any() else float("nan"),
        "referred": int((~auto).sum()),
    }

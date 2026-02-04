import math
import torch
import torch.nn.functional as F

class ScaleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.save_for_backward(scale)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        scale, = ctx.saved_tensors
        return grad_output * scale, None


def ecg_loss(logits, targets, lam=1.0, tau=0.7, k=10.0, conf_type="pmax", detach_gates=True, eps=1e-8):
    B, C = logits.shape

    p = F.softmax(logits, dim=1)
    py = p.gather(1, targets[:, None]).squeeze(1)

    ce = F.cross_entropy(logits, targets, reduction="none")

    # confidence
    if conf_type == "pmax":
        conf = p.max(dim=1).values
        conf_gate = torch.sigmoid(k * (conf - tau))
    elif conf_type == "1-pe":
        pe = -(p * (p.clamp_min(eps)).log()).sum(dim=1)
        pe_norm = pe / math.log(C)
        conf = 1.0 - pe_norm
        conf_gate = torch.sigmoid(k * (conf - tau))
    elif conf_type == "none":
        conf = torch.ones_like(py)  # dummy for logging
        conf_gate = torch.ones_like(py)
    else:
        raise ValueError("bad conf_type")

    wrong_gate = 1.0 - py

    if detach_gates:
        wrong_gate = wrong_gate.detach()
        conf_gate = conf_gate.detach()

    gate = wrong_gate * conf_gate
    scale = (1.0 + lam * gate).view(-1, 1)

    scaled_logits = ScaleGrad.apply(logits, scale)

    loss = F.cross_entropy(scaled_logits, targets)

    stats = {
        "gate_mean": gate.mean().item(),
        "wrong_mean": wrong_gate.mean().item(),
        "conf_mean": conf.mean().item(),
        "conf_gate_mean": conf_gate.mean().item(),
        "conf_gate_active_frac": (conf_gate > 0.5).float().mean().item(),
        "scale_mean": scale.mean().item(),
        "ce_mean": ce.mean().item(),
    }

    return loss, stats

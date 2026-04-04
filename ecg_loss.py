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


def ecg_loss(logits, targets, lam=1.0, tau=0.7, k=10.0, conf_type="pmax", detach_gates=True, eps=1e-8, tau_quantile=None, scale_normalize=False, gate_temp=1.5, minimal_stats=False):
    """
    tau_quantile: if set (e.g. 0.8), tau is set to this quantile of conf (pmax) per batch, reducing tau tuning.
    scale_normalize: if True, scale is normalized so mean(scale)=1 (for auto-lambda; keeps global step size unchanged).
    """
    B, C = logits.shape

    p = F.softmax(logits, dim=1)
    py = p.gather(1, targets[:, None]).squeeze(1)

    # confidence
    if conf_type == "pmax":
        conf = p.max(dim=1).values
        if tau_quantile is not None:
            tau = torch.quantile(conf.detach(), float(tau_quantile))
        conf_gate = torch.sigmoid(k * (conf - tau))
    elif conf_type == "1-pe":
        pe = -(p * (p.clamp_min(eps)).log()).sum(dim=1)
        pe_norm = pe / math.log(C)
        conf = 1.0 - pe_norm
        if tau_quantile is not None:
            tau = torch.quantile(conf.detach(), float(tau_quantile))
        conf_gate = torch.sigmoid(k * (conf - tau))
    elif conf_type == "margin":
        top2 = torch.topk(p, k=2, dim=1).values
        conf = top2[:, 0] - top2[:, 1]
        if tau_quantile is not None:
            tau = torch.quantile(conf.detach(), float(tau_quantile))
        conf_gate = torch.sigmoid(k * (conf - tau))
    elif conf_type == "pmax_temp":
        p_gate = F.softmax(logits / max(float(gate_temp), 1e-3), dim=1)
        conf = p_gate.max(dim=1).values
        if tau_quantile is not None:
            tau = torch.quantile(conf.detach(), float(tau_quantile))
        conf_gate = torch.sigmoid(k * (conf - tau))
    elif conf_type == "log_pmax":
        conf = -torch.log(1.0 - p.max(dim=1).values + eps)
        if tau_quantile is not None:
            tau = torch.quantile(conf.detach(), float(tau_quantile))
        conf_gate = torch.sigmoid(k * (conf - tau))
    elif conf_type == "logit_gap_norm":
        top2_logits = torch.topk(logits.detach().float(), k=2, dim=1).values
        raw_gap = top2_logits[:, 0] - top2_logits[:, 1]
        gap_mean = raw_gap.mean()
        gap_std = raw_gap.std(unbiased=False)
        conf = (raw_gap - gap_mean) / (gap_std + 1e-6)
        if tau_quantile is not None:
            tau = torch.quantile(conf, float(tau_quantile))
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
    if scale_normalize:
        scale = scale / (scale.mean().detach().clamp_min(1e-8))

    scaled_logits = ScaleGrad.apply(logits, scale)

    loss = F.cross_entropy(scaled_logits, targets)

    # --- control stats (always computed; used by auto-lambda / auto_tr) ---
    g_flat = gate.detach().float().view(-1)
    gate_mean_val = g_flat.mean().item()
    if g_flat.numel() <= 1:
        gate_p95_val = g_flat.item() if g_flat.numel() == 1 else 0.0
        gate_p99_val = gate_p95_val
    else:
        gate_p95_val = torch.quantile(g_flat, 0.95).item()
        gate_p99_val = torch.quantile(g_flat, 0.99).item()
    active_frac_val = (conf_gate > 0.5).float().mean().item()

    # conf histogram for auto_q_valley (50 fixed bins over [0, 1])
    _CONF_HIST_BINS = 50
    c_flat_hist = conf.detach().float().view(-1)
    conf_hist_counts = torch.histc(c_flat_hist, bins=_CONF_HIST_BINS, min=0.0, max=1.0)

    stats = {
        "gate_mean": gate_mean_val,
        "gate_p95": gate_p95_val,
        "gate_p99": gate_p99_val,
        "conf_gate_active_frac": active_frac_val,
        "_conf_hist": conf_hist_counts.cpu().tolist(),  # list of 50 floats; accumulated across batches
    }
    if scale_normalize:
        s_flat = scale.detach().float().view(-1)
        if s_flat.numel() <= 1:
            stats["scale_p99_after_norm"] = s_flat.item() if s_flat.numel() == 1 else 0.0
        else:
            stats["scale_p99_after_norm"] = torch.quantile(s_flat, 0.99).item()

    # --- logging-only stats (skipped when minimal_stats=True) ---
    if not minimal_stats:
        gate_std_val = g_flat.std().item() if g_flat.numel() > 1 else 0.0
        c_flat = conf.detach().float().view(-1)
        conf_p90_val = torch.quantile(c_flat, 0.90).item() if c_flat.numel() > 1 else c_flat.mean().item()
        conf_p95_val = torch.quantile(c_flat, 0.95).item() if c_flat.numel() > 1 else c_flat.mean().item()
        stats.update({
            "gate_std": gate_std_val,
            "wrong_mean": wrong_gate.mean().item(),
            "conf_mean": conf.mean().item(),
            "conf_p90": conf_p90_val,
            "conf_p95": conf_p95_val,
            "tau_threshold": float(tau) if not isinstance(tau, torch.Tensor) else tau.item(),
            "ecg_gate_temp": float(gate_temp) if conf_type == "pmax_temp" else 0.0,
            "logit_gap_mean": gap_mean.item() if conf_type == "logit_gap_norm" else 0.0,
            "logit_gap_std": gap_std.item() if conf_type == "logit_gap_norm" else 0.0,
            "conf_gate_mean": conf_gate.mean().item(),
            "scale_mean": scale.mean().item(),
            "ce_mean": loss.item(),
        })
        if scale_normalize:
            stats["scale_std_after_norm"] = scale.detach().std().item() if scale.numel() > 1 else 0.0

    return loss, stats


def ecg_gates(logits, targets, lam=1.0, tau=0.7, k=10.0, conf_type="pmax", detach_gates=True, eps=1e-8, gate_temp=1.5):
    """
    Compute per-sample gate/scale stats for visualization (no grad needed).

    Returns dict of 1D tensors (shape [B]).
    """
    import math
    import torch
    import torch.nn.functional as F

    B, C = logits.shape
    p = F.softmax(logits, dim=1)
    py = p.gather(1, targets[:, None]).squeeze(1)

    # confidence
    if conf_type == "pmax":
        conf = p.max(dim=1).values
        conf_gate = torch.sigmoid(k * (conf - tau))
    elif conf_type == "margin":
        top2p = torch.topk(p, k=2, dim=1).values
        conf = top2p[:, 0] - top2p[:, 1]
        conf_gate = torch.sigmoid(k * (conf - tau))
    elif conf_type == "pmax_temp":
        p_gate = F.softmax(logits / max(float(gate_temp), 1e-3), dim=1)
        conf = p_gate.max(dim=1).values
        conf_gate = torch.sigmoid(k * (conf - tau))
    elif conf_type == "1-pe":
        pe = -(p * (p.clamp_min(eps)).log()).sum(dim=1)
        pe_norm = pe / math.log(C)
        conf = 1.0 - pe_norm
        conf_gate = torch.sigmoid(k * (conf - tau))
    elif conf_type == "log_pmax":
        conf = -torch.log(1.0 - p.max(dim=1).values + eps)
        conf_gate = torch.sigmoid(k * (conf - tau))
    elif conf_type == "logit_gap_norm":
        top2_logits = torch.topk(logits.float(), k=2, dim=1).values
        raw_gap = top2_logits[:, 0] - top2_logits[:, 1]
        gap_mean = raw_gap.mean()
        gap_std = raw_gap.std(unbiased=False)
        conf = (raw_gap - gap_mean) / (gap_std + 1e-6)
        conf_gate = torch.sigmoid(k * (conf - tau))
    elif conf_type == "none":
        conf = torch.ones_like(py)
        conf_gate = torch.ones_like(py)
    else:
        raise ValueError("bad conf_type")

    wrong_gate = 1.0 - py
    if detach_gates:
        wrong_gate = wrong_gate.detach()
        conf_gate = conf_gate.detach()

    gate = wrong_gate * conf_gate
    scale = (1.0 + lam * gate)

    # probability margin (top1 - top2 prob)
    top2p = torch.topk(p, k=2, dim=1).values
    margin = top2p[:, 0] - top2p[:, 1]

    return {
        "py": py.detach(),
        "pmax": p.max(dim=1).values.detach(),
        "margin": margin.detach(),
        "gate": gate.detach(),
        "scale": scale.detach(),
    }

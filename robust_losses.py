"""Robust-training baselines: focal, CLUE-lite, CLUE, PGD-AT, TRADES, MART."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss(logits, targets, gamma=2.0, alpha=1.0):
    ce_i = F.cross_entropy(logits, targets, reduction="none")
    pt = F.softmax(logits, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
    factor = (1.0 - pt).pow(gamma)
    loss = (alpha * factor * ce_i).mean()
    return loss, {
        "focal_pt_mean": pt.mean().item(),
        "focal_factor_mean": factor.mean().item(),
        "ce_mean": ce_i.mean().item(),
    }


def clue_lite_loss(logits, targets, clue_lambda=0.2, detach_proxy=True, eps=1e-8):
    p = F.softmax(logits, dim=1)
    C = logits.shape[1]
    log_C = torch.log(torch.tensor(float(C), device=logits.device))
    ce_i = F.cross_entropy(logits, targets, reduction="none")
    entropy = -(p * (p.clamp_min(eps)).log()).sum(dim=1) / log_C
    error_proxy = (ce_i / log_C).clamp(0.0, 1.5)
    align_target = error_proxy.detach() if detach_proxy else error_proxy
    align = F.mse_loss(entropy, align_target)
    loss = ce_i.mean() + clue_lambda * align
    return loss, {
        "clue_entropy_mean": entropy.mean().item(),
        "clue_proxy_mean": error_proxy.mean().item(),
        "clue_align": align.item(),
        "ce_mean": ce_i.mean().item(),
    }


def pgd_attack_ce(model, x, y, epsilon, alpha, steps, random_start=True):
    """PGD-Linf maximizing CE. Pixel space [0,1]."""
    x_adv = x.detach().clone()
    if random_start:
        x_adv = x_adv + torch.zeros_like(x_adv).uniform_(-epsilon, epsilon)
        x_adv = x_adv.clamp(0.0, 1.0)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        delta = (x_adv - x).clamp(-epsilon, epsilon)
        x_adv = (x + delta).clamp(0.0, 1.0)
    return x_adv.detach()


def pgd_attack_trades(model, x, logits_nat_detached, epsilon, alpha, steps,
                      random_start=True):
    """PGD-Linf maximizing KL for TRADES. Pixel space [0,1]."""
    p_nat = F.softmax(logits_nat_detached, dim=1)
    x_adv = x.detach().clone()
    if random_start:
        x_adv = x_adv + torch.zeros_like(x_adv).uniform_(-epsilon, epsilon)
        x_adv = x_adv.clamp(0.0, 1.0)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits_adv = model(x_adv)
        loss_kl = F.kl_div(
            F.log_softmax(logits_adv, dim=1), p_nat, reduction="batchmean"
        )
        grad = torch.autograd.grad(loss_kl, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        delta = (x_adv - x).clamp(-epsilon, epsilon)
        x_adv = (x + delta).clamp(0.0, 1.0)
    return x_adv.detach()


def trades_loss(model, x, y, epsilon, alpha, steps, beta=6.0, random_start=True):
    """TRADES (Zhang et al. 2019): CE_nat + beta * KL(adv || nat)."""
    logits_nat = model(x)
    x_adv = pgd_attack_trades(
        model, x, logits_nat.detach(), epsilon, alpha, steps, random_start
    )
    logits_adv = model(x_adv)
    loss_nat = F.cross_entropy(logits_nat, y)
    loss_robust = F.kl_div(
        F.log_softmax(logits_adv, dim=1),
        F.softmax(logits_nat.detach(), dim=1),
        reduction="batchmean",
    )
    loss = loss_nat + beta * loss_robust
    return loss, {
        "trades_loss_nat": loss_nat.item(),
        "trades_loss_robust": loss_robust.item(),
        "trades_beta": beta,
    }


def mart_loss(model, x, y, epsilon, alpha, steps, beta=6.0, random_start=True):
    """MART (Wang et al. 2020): misclassification-aware robust training."""
    x_adv = pgd_attack_ce(model, x, y, epsilon, alpha, steps, random_start=random_start)
    logits = model(x)
    logits_adv = model(x_adv)
    adv_probs = F.softmax(logits_adv, dim=1)
    tmp = adv_probs.argsort(dim=1)[:, -2:]
    new_y = torch.where(tmp[:, -1] == y, tmp[:, -2], tmp[:, -1])
    loss_adv = F.cross_entropy(logits_adv, y) + F.nll_loss(
        torch.log(1.0001 - adv_probs + 1e-12), new_y
    )
    nat_probs = F.softmax(logits, dim=1)
    true_probs = nat_probs.gather(1, y.unsqueeze(1)).squeeze(1)
    kl_per_sample = F.kl_div(
        torch.log(adv_probs + 1e-12), nat_probs, reduction="none"
    ).sum(dim=1)
    loss_robust = (kl_per_sample * (1.0000001 - true_probs)).mean()
    loss = loss_adv + beta * loss_robust
    return loss, {
        "mart_loss_adv": loss_adv.item(),
        "mart_loss_robust": loss_robust.item(),
        "mart_true_prob_mean": true_probs.mean().item(),
        "mart_beta": beta,
    }


def _enable_dropout(model):
    """Enable dropout layers for MC inference while keeping BN in eval mode."""
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d)):
            m.train()


def mc_dropout_uncertainty(model, x, mc_passes=5, eps=1e-8):
    """Estimate predictive entropy via MC dropout.

    Keeps BN in eval mode, only enables dropout for stochastic passes.
    Returns per-sample predictive entropy [B].
    """
    was_training = model.training
    model.eval()
    _enable_dropout(model)

    probs_sum = None
    with torch.no_grad():
        for _ in range(mc_passes):
            logits = model(x)
            p = F.softmax(logits, dim=1)
            if probs_sum is None:
                probs_sum = p
            else:
                probs_sum = probs_sum + p
    p_mean = probs_sum / mc_passes
    ent = -(p_mean * (p_mean + eps).log()).sum(dim=1)

    model.train(was_training)
    return ent


def clue_loss(model, x, y, alpha=0.5, mc_passes=5, eps=1e-8):
    """Classification CLUE loss: alpha * L_e + (1-alpha) * (L_e - u)^2.

    L_e = per-sample CE loss (with gradient)
    u   = predictive entropy from MC dropout (detached, no gradient)
    """
    uncertainty = mc_dropout_uncertainty(model, x, mc_passes=mc_passes, eps=eps)
    u = uncertainty.detach()

    logits = model(x)
    ce_i = F.cross_entropy(logits, y, reduction="none")

    align = (ce_i - u).pow(2)
    loss = (alpha * ce_i + (1.0 - alpha) * align).mean()

    return loss, {
        "clue_ce_mean": ce_i.mean().item(),
        "clue_unc_mean": u.mean().item(),
        "clue_unc_std": u.std().item(),
        "clue_align_mean": align.mean().item(),
        "clue_alpha": alpha,
        "clue_mc_passes": mc_passes,
    }

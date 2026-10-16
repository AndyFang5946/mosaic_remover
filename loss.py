import torch
import torch.nn.functional as F

def masked_l1(pred, target, mask, eps=1e-8):
    mask = mask.expand_as(pred)
    error = (torch.abs(pred - target) * mask)
    return error.sum()/(mask.sum() + eps)

def _ssim_map(x, y, window_size=11, c1=0.01**2, c2=0.03**2):
    padding = window_size // 2

    mu_x = F.avg_pool2d(x, window_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(y, window_size, stride=1, padding=padding)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.avg_pool2d(x * x, window_size, stride=1, padding=padding) - mu_x2
    sigma_y2 = F.avg_pool2d(y * y, window_size, stride=1, padding=padding) - mu_y2
    sigma_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=padding) - mu_xy

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)

    return torch.clamp(numerator/(denominator + 1e-8), -1.0, 1.0)

def masked_ssim_loss(pred, mask, target, window_size=11, eps=1e-8):
    ssim = _ssim_map(pred, target, window_size)
    mask = mask.expand_as(pred)

    loss = (1.0 - ssim) * mask

    return loss.sum()/(mask.sum() + eps)

def reconstruction_loss(pred, target, mask, ssim_weight=0.2):
    l1 = masked_l1(pred, target, mask)
    ssim = masked_ssim_loss(pred, target, mask)
    total = (1.0 - ssim_weight)*l1 + ssim_weight * ssim
    return total, l1.detach(), ssim.detach()

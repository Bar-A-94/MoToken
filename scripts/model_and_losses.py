import torch
from torch import nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.1)
        x = self.fc2(x)
        return x.flatten()


def sorted_l1_loss(alphas):
    values, _ = torch.sort(torch.abs(alphas), descending=True)
    weights = torch.arange(1, len(values) + 1, device=alphas.device).float()
    weights = weights / weights.sum()
    return (values * weights).sum()


def calculate_motion_loss(pred_latents, orig_latents, frame_diff):
    pred_diff = torch.abs(pred_latents[:, frame_diff:] - pred_latents[:, :-frame_diff])   # [B, F-diff, H, W, C]
    orig_diff = torch.abs(orig_latents[:, frame_diff:] - orig_latents[:, :-frame_diff])   # [B, F-diff, H, W, C]
    return F.mse_loss(pred_diff.float(), orig_diff.float(), reduction="mean")

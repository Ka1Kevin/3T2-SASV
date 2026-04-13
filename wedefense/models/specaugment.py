import torch
import torch.nn as nn
from typing import Tuple
from torch.jit import Final

class SpecAugment(nn.Module):
    def __init__(self, freq_mask_width: Tuple[int, int] = (0, 8), time_mask_width: Tuple[int, int] = (0, 10)):
    # def __init__(self, freq_mask_width = (0, 8), time_mask_width = (0, 10)):
        # self.time_mask_width = time_mask_width
        # self.freq_mask_width = freq_mask_width
        super(SpecAugment, self).__init__()
        self.freq_mask_width: Final[Tuple[int, int]] = freq_mask_width
        self.time_mask_width: Final[Tuple[int, int]] = time_mask_width

    @torch.jit.export
    def mask_along_axis(self, x: torch.Tensor, dim: int) -> torch.Tensor:
    # def mask_along_axis(self, x, dim):
        # original_size = x.shape
        # batch, fea, time = x.shape
        B = x.size(0)
        F = x.size(1)
        T = x.size(2)
        if dim == 1:
            D = F
            width_low, width_high = self.freq_mask_width
        else:
            D = T
            width_low, width_high = self.time_mask_width
            # width_range = self.time_mask_width
        
        mask_len = torch.randint(low=width_low, high=width_high+1, size=(B,), device=x.device)
        # mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        max_start = torch.clamp(D - mask_len, min=1)
        mask_pos = torch.randint(low=0, high=int(max_start.max()), size=(B,), device=x.device)
        # mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        # arange = torch.arange(D, device=x.device).view(1, 1, -1)
        # mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        # mask = mask.any(dim=1)
        arange = torch.arange(D, device=x.device).unsqueeze(0).expand(B, -1)
        mask = (arange >= mask_pos.unsqueeze(1)) & (arange < (mask_pos + mask_len).unsqueeze(1))

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)

        return x.masked_fill(mask, 0.0)
        # x = x.masked_fill_(mask, 0.0)
        # return x.view(*original_size)

    def forward(self, x):
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x
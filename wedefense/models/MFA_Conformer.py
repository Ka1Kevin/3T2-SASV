import torch
import torch.nn as nn
import torchaudio
from torch import Tensor
from typing import Tuple
from .specaugment import SpecAugment
from .wenet.transformer.encoder_cat import ConformerEncoder

class Conformer(nn.Module):
    def __init__(self, num_mels=80, num_blocks=6, output_size=256, embedding_dim=192, input_layer="conv2d2", pos_enc_layer_type="rel_pos"):
        super(Conformer, self).__init__()
        print("input_layer: {}".format(input_layer))
        print("pos_enc_layer_type: {}".format(pos_enc_layer_type))
        self.conformer = ConformerEncoder(input_size=num_mels, num_blocks=num_blocks, output_size=output_size, input_layer=input_layer, pos_enc_layer_type=pos_enc_layer_type, )
        self.bn = nn.BatchNorm1d(output_size*num_blocks*2)
        self.fc = nn.Linear(output_size*num_blocks*2, embedding_dim)

        self.specaug = SpecAugment()
        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80),
            )
        output_dim = output_size*num_blocks
        self.attention = nn.Sequential(
            nn.Conv1d(output_dim*3, 256, kernel_size=1),
            nn.ReLU(inplace=False),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Conv1d(256, output_dim, kernel_size=1),
            nn.Softmax(dim=2),
            )

    def forward(self, x: Tensor, aug: bool = False) -> Tensor:
    # def forward(self, x: Tensor, aug=False) -> Tuple[Tensor, bool]:

        with torch.no_grad():
            with torch.amp.autocast("cuda",enabled=False):
            # with torch.cuda.amp.autocast("cuda",enabled=False):
                if hasattr(self, "frontend"):  
                    x = self.frontend(x)        
                else:
                    x = self.torchfbank(x)+1e-6 # (B, F, T) (32, 80, 501)
                    x = x.log() # (32, 80, 501)
                    # x = x - torch.mean(x, dim=-1, keepdim=True)
                    # x = x - torch.mean(x, dim=-1, keepdim=True).detach()
                    mean = x.mean(dim=2, keepdim=True) # (32, 80, 1)
                    std = x.std(dim=2, keepdim=True, unbiased=False).clamp(min=1e-5) # (16, 80, 1)
                    x = (x - mean) / std
                    if aug == True:
                        x = self.specaug(x)
        x = x.transpose(1,2).contiguous()  # (B, T, F) (32, 501, 80)
        lens = torch.ones(x.shape[0]).to(x.device)
        lens = torch.round(lens*x.shape[1]).int() # (32)
        x, masks = self.conformer(x, lens) # (32, 250, 1536)
        x = x.transpose(1,2).contiguous() # (32, 1536, 250)

        # Context dependent ASP
        t = x.size()[-1] # 250
        mean_x = torch.mean(x, dim=2, keepdim=True)
        var_x = torch.var(x, dim=2, keepdim=True, unbiased=False)
        std_x = torch.sqrt(torch.clamp(var_x, min=1e-4))

        global_x = torch.cat(
            [x, mean_x.expand(-1, -1, t).clone(), std_x.expand(-1, -1, t).clone()], dim=1
        ).contiguous() # (32, 4608, 250)
        # global_x = torch.cat((x,torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t), torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t)), dim=1)
        w = self.attention(global_x) # (32, 1536, 250)
        mu = torch.sum(x * w, dim=2) # (32, 1536)
        # sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )
        second_moment = torch.sum(x.pow(2) * w, dim=2)
        sg = torch.sqrt(torch.clamp(second_moment - mu.pow(2), min=1e-4))

        x = torch.cat((mu, sg), dim=1) # (32, 3072)
        # BN -> FC: embedding
        # x = self.bn(x.clone())
        # x = self.fc(x)
        # return x
        # 拼接后的特征 (B, D_total)
        x = x.contiguous()             # 确保连续内存
        x = self.bn(x) # (32, 3072)
        x = self.fc(x) # (32, 192)
        return x
    
class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        # make kernel
        # In pytorch, the convolution operation uses cross-correlation. So, filter is flipped.
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.conv1d 需要输入 shape [B, C, T]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return torch.nn.functional.conv1d(x, self.flipped_filter, padding=1).squeeze(1)

def MFA_Conformer(num_mels=80, num_out=192, **kwargs):
    model = Conformer(num_mels=num_mels, embedding_dim=num_out, input_layer="conv2d2")
    return model

# MFA_Conformer = MainModel
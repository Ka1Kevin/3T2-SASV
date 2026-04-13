# Copyright (c) 2021 Hongji Wang (jijijiang77@gmail.com)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os
import fire
import kaldiio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import sys
sys.path.append(os.path.abspath('.'))

from wedefense.dataset.dataset_new import Dataset
from wedefense.dataset.dataset_utils import apply_cmvn, spec_aug
from wedefense.frontend import *
from wedefense.models.get_model import get_model
from wedefense.utils.checkpoint import load_checkpoint
from wedefense.utils.utils import parse_config_or_kwargs, validate_path

def multiwave_collate(batch):
    keys = [item["key"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

    refs_list = []
    test_list = []
    for item in batch:
        wavs = item["wav"]
        refs, test = wavs[:-1], wavs[-1]
        refs_list.append(refs)
        test_list.append(test)

    n_ref = len(refs_list[0])

    refs_per_idx = [[] for _ in range(n_ref)]
    for refs in refs_list:
        for i, r in enumerate(refs):
            refs_per_idx[i].append(r)

    refs_padded = []
    for i in range(n_ref):
        refs_padded.append(pad_sequence(refs_per_idx[i], batch_first=True))  # (B, T_i)
    refs_batch = torch.stack(refs_padded, dim=1)  # (B, N_ref, T_max)

    test_batch = pad_sequence(test_list, batch_first=True)  # (B, T_test)

    return {
        "keys": keys,          # list[str]
        "refs": refs_batch,    # (B, N_ref, T)
        "test": test_batch,    # (B, T)
        "labels": labels       # (B,)
    }

class MHACrossAttnAdd(nn.Module):
    """
    Same as training: MultiheadAttention(test as query, refs as key/value), then residual add + LayerNorm.
    """
    def __init__(self, embed_dim=192, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, test_embed, refs_embed):
        # test_embed: (B,D), refs_embed: (B,N,D)
        q = test_embed.unsqueeze(1)          # (B,1,D)
        ctx, _ = self.attn(q, refs_embed, refs_embed)  # (B,1,D)
        ctx = ctx.squeeze(1)                # (B,D)
        return self.ln(test_embed + ctx)    # (B,D)

def build_embeds_extract(model, test, refs, configs):
    """
    Reproduce training-time build_embeds() for extraction.

    model: non-DDP model in extract
    test: (B,T)
    refs: (B,N_refs,T)
    return embeds (B, D) for xattn_add/time_cat, or (B, (1+N)*D) for feat_cat
    """
    fusion_type = configs.get('fusion_type', 'xattn_add')
    B, N_refs, T = refs.shape

    if fusion_type == 'feat_cat':
        # forward per-utt then flatten
        all_wavs = torch.cat([test.unsqueeze(1), refs], dim=1)   # (B,1+N,T)
        all_wavs = all_wavs.reshape(B * (1 + N_refs), T)         # (B*(1+N),T)
        all_embeds = model(all_wavs)                             # (B*(1+N),D)
        all_embeds = all_embeds.reshape(B, 1 + N_refs, -1)       # (B,1+N,D)
        embeds = all_embeds.reshape(B, -1)                       # (B,(1+N)*D)
        return embeds

    elif fusion_type == 'xattn_add':
        # need fusion_module on model
        fusion = getattr(model, "fusion_module", None)
        if fusion is None:
            raise RuntimeError(
                "fusion_type='xattn_add' but model.fusion_module is missing in extract. "
                "Make sure you add it before load_checkpoint()."
            )

        refs_flat = refs.reshape(B * N_refs, T)                 # (B*N, T)
        all_wav = torch.cat([test, refs_flat], dim=0)           # (B + B*N, T)
        all_emb = model(all_wav)                                # (B + B*N, D)
        test_embed = all_emb[:B]                                # (B, D)
        refs_embed = all_emb[B:].reshape(B, N_refs, -1)         # (B, N, D)
        embeds = fusion(test_embed, refs_embed)                 # (B, D)
        return embeds

    elif fusion_type == 'time_cat':
        # waveform-level time concat, forward once
        refs_time = refs.reshape(B, N_refs * T)                 # (B, N*T)
        wav_cat = torch.cat([test, refs_time], dim=1)           # (B, (N+1)*T)
        embeds = model(wav_cat)                                 # (B, D)
        return embeds

    else:
        raise ValueError(f"Unknown fusion_type: {fusion_type}")

def extract(config='conf/config.yaml', **kwargs):
    # parse configs first
    configs = parse_config_or_kwargs(config, **kwargs)

    model_path = configs['model_path']
    embed_ark = configs['embed_ark']
    batch_size = configs.get('batch_size', 1)
    num_workers = configs.get('num_workers', 1)

    # Since the input length is not fixed, we set the built-in cudnn
    # auto-tuner to False
    torch.backends.cudnn.benchmark = False

    test_conf = copy.deepcopy(configs['dataset_args'])
    # model: frontend (optional) => speaker model
    model = get_model(configs['model'])(**configs['model_args'])

    fusion_type = configs.get('fusion_type', 'xattn_add')
    if fusion_type == 'xattn_add':
        model.add_module(
            "fusion_module",
            MHACrossAttnAdd(
                embed_dim=configs['model_args']['embed_dim'],
                num_heads=configs.get('xattn_heads', 4),
                dropout=configs.get('xattn_dropout', 0.1),
            )
        )

    frontend_type = test_conf.get('frontend', 'fbank')
    if frontend_type != "fbank" and not frontend_type.startswith('lfcc'):
        frontend_args = frontend_type + "_args"
        # frontends besides acoustic features, like s3prl
        print('Initializing frontend model (this could take some time) ...')
        frontend = frontend_class_dict[frontend_type](
            **test_conf[frontend_args], sample_rate=test_conf['resample_rate'])
        model.add_module("frontend", frontend)
    print('Loading checkpoint ...')
    load_checkpoint(model, model_path)
    print('Finished !!! Start extracting ...')
    device = torch.device("cuda")
    # device = torch.device("cpu")
    model.to(device).eval()

    # test_configs
    # test_conf = copy.deepcopy(configs['dataset_args'])
    test_conf['speed_perturb'] = False
    if 'fbank_args' in test_conf:
        test_conf['fbank_args']['dither'] = 0.0
    test_conf['spec_aug'] = False
    test_conf['shuffle'] = False
    test_conf['aug_prob'] = configs.get('aug_prob', 0.0)
    test_conf['filter'] = False
    test_conf['codec_aug'] = False
    test_conf['rawboost'] = False

    dataset = Dataset(configs['data_type'],
                      configs['data_list'],
                      test_conf,
                      lab2id_dict={},
                      whole_utt=(batch_size == 1),
                      reverb_lmdb_file=configs.get('reverb_data', None),
                      noise_lmdb_file=configs.get('noise_data', None),
                      repeat_dataset=False)
    
    dataloader = DataLoader(dataset,
                            shuffle=False,
                            collate_fn=multiwave_collate,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            prefetch_factor=4)

    validate_path(embed_ark)
    embed_ark = os.path.abspath(embed_ark)
    embed_scp = embed_ark[:-3] + "scp"

    base_dim = configs['model_args']['embed_dim']
    fusion_type = configs.get('fusion_type', 'xattn_add')
    print(f"[extract] fusion_type={fusion_type}, base_dim={base_dim}")


    with torch.no_grad():
        with kaldiio.WriteHelper('ark,scp:' + embed_ark + "," +
                                 embed_scp) as writer:
            for _, batch in tqdm(enumerate(dataloader)):
                test = batch['test'].to(device).float()
                refs = batch['refs'].to(device).float()
                B, N_ref, T = refs.shape
                if fusion_type == 'feat_cat':
                    expected_dim = base_dim * (1 + N_ref)
                elif fusion_type in ['xattn_add', 'time_cat']:
                    expected_dim = base_dim
                else:
                    raise ValueError(f"Unknown fusion_type: {fusion_type}")

                with torch.amp.autocast('cuda', enabled=configs.get('enable_amp', False)):
                    embeds = build_embeds_extract(model, test, refs, configs)  # (B,D) or (B,(1+N)*D)

                # strict check: must match training-time projection input dim
                if embeds.size(1) != expected_dim:
                    raise RuntimeError(
                        f"[extract] embeds={tuple(embeds.shape)} but projection expects {expected_dim}. "
                        f"fusion_type={fusion_type}, B={B}, N_ref={N_ref}, T={T}"
                    )

                embeds_np = embeds.cpu().numpy()

                for i in range(B):
                    writer(batch['keys'][i], embeds_np[i])
    writer.close()
    print("Done writing embeddings.")



if __name__ == '__main__':
    fire.Fire(extract)

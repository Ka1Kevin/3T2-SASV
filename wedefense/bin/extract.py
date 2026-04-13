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
from torch.utils.data import DataLoader
from tqdm import tqdm

from wedefense.dataset.dataset import Dataset
from wedefense.dataset.dataset_utils import apply_cmvn, spec_aug
from wedefense.frontend import *
from wedefense.models.get_model import get_model
from wedefense.utils.checkpoint import load_checkpoint
from wedefense.utils.utils import parse_config_or_kwargs, validate_path


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
                            batch_size=batch_size,
                            num_workers=num_workers,
                            prefetch_factor=4)

    validate_path(embed_ark)
    embed_ark = os.path.abspath(embed_ark)
    embed_scp = embed_ark[:-3] + "scp"

    with torch.no_grad():
        with kaldiio.WriteHelper('ark,scp:' + embed_ark + "," +
                                 embed_scp) as writer:
            for _, batch in tqdm(enumerate(dataloader)):
                targets = batch['labels']
                B = targets.shape[0]
                concat_feats = []
                for b in range(B):
                    ref_feat_list = []
                    if frontend_type == 'fbank' or frontend_type.startswith('lfcc'):
                        # batch['ref_feats'][b] 形状: (N_ref, T, F)
                        ref_feat_list = [batch['ref_feats'][b][j].to(device).float()
                                        for j in range(batch['ref_feats'][b].shape[0])]
                    else:  # 's3prl'
                        # batch['refs'][b] 形状: (N_ref, W)
                        for j in range(batch['refs'][b].shape[0]):
                            wav = batch['refs'][b][j].unsqueeze(0).float().to(device)  # (1,80000)
                            wav_len = torch.LongTensor([wav.shape[1]]).to(device)
                            with torch.cuda.amp.autocast(enabled=configs['enable_amp']):
                                feat, _ = model.module.frontend(wav, wav_len)  # (1,T,F)
                                # feat, _ = model.frontend(wav, wav_len) # (1, 768, 250, 13)
                            ref_feat_list.append(feat.squeeze(0))
                    if len(ref_feat_list) > 0:
                        ref_concat = torch.cat(ref_feat_list, dim=0)  # (768, 250, 13)
                    else:
                        ref_concat = torch.empty(0).to(device)

                    if frontend_type == 'fbank' or frontend_type.startswith('lfcc'):
                        test_feat = batch['feat'][b].to(device).float()  # 
                    else:  # 's3prl'
                        test_wav = batch['test'][b].unsqueeze(0).float().to(device)  # 
                        wav_len = torch.LongTensor([test_wav.shape[1]]).to(device)
                        with torch.cuda.amp.autocast(enabled=configs['enable_amp']):
                            test_feat, _ = model.module.frontend(test_wav, wav_len)  # (1,T,F)
                            # test_feat, _ = model.frontend(test_wav, wav_len) # (1, 768, 250, 13)
                        test_feat = test_feat.squeeze(0)
                    concat_feat = torch.cat([ref_concat, test_feat], dim=0)  # (1536, 250, 13)
                    concat_feats.append(concat_feat)
                features = pad_sequence(concat_feats, batch_first=True)  # (128, 1536, 250, 13)

                # apply cmvn
                if test_conf.get('cmvn', True):
                    features = apply_cmvn(features,
                                          **test_conf.get('cmvn_args', {}))
                # spec augmentation
                if test_conf.get('spec_aug', False):
                    features = spec_aug(features, **test_conf['spec_aug_args'])

                # Forward through model
                outputs = model(features)  # embed or (embed_a, embed_b)
                embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
                embeds = embeds.cpu().detach().numpy()  # (B,F)

                for i, utt in enumerate(utts):
                    embed = embeds[i]
                    writer(utt, embed)


if __name__ == '__main__':
    fire.Fire(extract)

# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang)
#               2021 Hongji Wang (jijijiang77@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import logging


def load_checkpoint(model: torch.nn.Module, path: str):
    checkpoint = torch.load(path, map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint,
                                                          strict=False)
    for key in missing_keys:
        logging.warning('missing tensor: {}'.format(key))
    for key in unexpected_keys:
        logging.warning('unexpected tensor: {}'.format(key))

def load_checkpoint_new(model, path: str):
    print("[load_checkpoint_new] called:", path)

    ckpt = torch.load(path, map_location="cpu")

    # 取出真正的 state_dict（按常见格式逐个试）
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state = ckpt["model"]
        else:
            state = ckpt  # 兜底：本身就是 state_dict
    else:
        state = ckpt

    # strip 前缀
    new_state = {}
    for k, v in state.items():
        k2 = k
        if k2.startswith("__S__."):
            k2 = k2[len("__S__."):]
        if k2.startswith("module."):
            k2 = k2[len("module."):]
        new_state[k2] = v

    # 验证 strip 是否成功（看前几条 key）
    keys = list(new_state.keys())
    print("[load_checkpoint_new] example keys:", keys[:5])

    missing, unexpected = model.load_state_dict(new_state, strict=False)

    print(f"[load_checkpoint_new] missing={len(missing)}, unexpected={len(unexpected)}")
    for key in missing[:20]:
        logging.warning("missing tensor: %s", key)
    for key in unexpected[:20]:
        logging.warning("unexpected tensor: %s", key)



def save_checkpoint(model: torch.nn.Module, path: str):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(state_dict, path)

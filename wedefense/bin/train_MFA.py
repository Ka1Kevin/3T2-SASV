# Copyright (c) 2021 Hongji Wang (jijijiang77@gmail.com)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#               2025 Lin Zhang (partialspoof@gmail.com)
#                    Shuai Wang (wsstriving@gmail.com)
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

import os
import re
from pprint import pformat
import sys
import fire
import tableprint as tp
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader
sys.path.append(os.path.abspath('.'))
import wedefense.utils.schedulers as schedulers
from wedefense.dataset.dataset_new import Dataset, DataList, Processor, parse_multi_raw
from wedefense.frontend import *
from wedefense.models.projections import get_projection
from wedefense.models.get_model import get_model
from wedefense.utils.checkpoint import load_checkpoint_new, save_checkpoint
from wedefense.utils.executor_MFA_CE import train_epoch, val_epoch, MHACrossAttnAdd
from wedefense.utils.file_utils import read_table, read_lists
from wedefense.utils.utils import get_logger, parse_config_or_kwargs, set_seed, \
    lab2id
from torch.nn.utils.rnn import pad_sequence
import wedefense.dataset.customize_collate_fn as nii_collate_fn
import torch.nn as nn
import random
from torch.utils.data import Subset
from wedefense.dataset import processor
import wandb
import time


def multiwave_collate(batch):
    keys = [item["key"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

    # 拆分 ref 和 test
    refs_list = []
    test_list = []
    for item in batch:
        wavs = item["wav"]
        refs, test = wavs[:-1], wavs[-1]
        refs_list.append(refs)
        test_list.append(test)

    # 确定参考语音数量（固定）
    n_ref = len(refs_list[0])

    # 对每个 ref 索引分开 pad，再 stack
    refs_per_idx = [[] for _ in range(n_ref)]
    for refs in refs_list:
        for i, r in enumerate(refs):
            refs_per_idx[i].append(r)

    refs_padded = []
    for i in range(n_ref):
        refs_padded.append(pad_sequence(refs_per_idx[i], batch_first=True))  # (B, T_i)
    refs_batch = torch.stack(refs_padded, dim=1)  # (B, N_ref, T_max)

    # padding test audio
    test_batch = pad_sequence(test_list, batch_first=True)  # (B, T_test)

    return {
        "keys": keys,          # list[str]
        "refs": refs_batch,    # (B, N_ref, T)
        "test": test_batch,    # (B, T)
        "labels": labels       # (B,)
    }

def train(config='conf/config.yaml', **kwargs):
    """Trains a model on the given features and lab labels.

    :config: A training configuration. Note that all parameters in the
             config can also be manually adjusted with --ARG VALUE
    :returns: None
    """
    configs = parse_config_or_kwargs(config, **kwargs)

    # wandb info
    model_dir = os.getcwd()  # current dir: model dir
    egs_dir = model_dir.split(
        '/egs/')[0] + '/egs'  # /path/to/<task>/<database>/<model>

    project_name = f"wedefense/{os.path.relpath(model_dir, egs_dir)}"  # wedefense/<task>/<database>/<model>
    model_name = os.path.basename(project_name)  # <model>
    run_name = f"{model_name}/{configs['exp_dir']}_{time.strftime('%Y%m%d_%H%M%S')}".replace(
        "exp/", "")

    wandb_run = wandb.init(
        project=os.path.dirname(project_name).replace(os.sep, "_"),
        name=run_name,
        config={
            "model": configs['model'],
        },
    )

    checkpoint = configs.get('checkpoint', None)
    # dist configs
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ['WORLD_SIZE'])
    # gpu = int(configs['gpus'].strip('[]').split(',')[local_rank])
    gpu = int(configs['gpus'][local_rank]) 
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl')
    device = torch.device(f"cuda:{gpu}")
    

    model_dir = os.path.join(configs['exp_dir'], "models")
    if rank == 0:
        try:
            os.makedirs(model_dir, exist_ok=True)
        except IOError:
            print("[warning] " + model_dir + " already exists !!!")
            if checkpoint is None:
                print("[error] checkpoint is null !")
                exit(1)
    dist.barrier(device_ids=[gpu])  # let the rank 0 mkdir first 

    logger = get_logger(configs['exp_dir'], 'train.log')
    if world_size > 1:
        logger.info('training on multiple gpus, this gpu {}'.format(gpu))

    if rank == 0:
        logger.info("exp_dir is: {}".format(configs['exp_dir']))
        logger.info("<== Passed Arguments ==>")
        # Print arguments into logs
        for line in pformat(configs).split('\n'):
            logger.info(line)

    # seed
    set_seed(configs['seed'] + rank)

    # train data
    train_label = configs['train_label']
    train_utt_lab_list = read_table(train_label) # ["spk#utt", "label"]
    lab2id_dict = lab2id(train_utt_lab_list)
    if rank == 0:
        logger.info("<== Data statistics ==>")
        logger.info("train data num: {}, class num: {}".format(
            len(train_utt_lab_list), len(lab2id_dict)))

    batch_size = configs['dataloader_args']['batch_size']
    whole_utt = configs['dataset_args'].get(
        'whole_utt', False)  # set as True after debugging.
    sampler = configs['dataset_args'].get('sampler', None)

    # collate function
    multiwav = True
    if whole_utt and batch_size > 1:
        # for batch-size > 1, use customize_collate to handle
        # data with different length
        collate_fn = nii_collate_fn.customize_collate
    elif multiwav:
        collate_fn = multiwave_collate
    else:
        collate_fn = None

    # TODO sampler to support building mini-batch according to length.
    tmp_params_dataloader = configs['dataloader_args'].copy()
    if sampler == 'block_shuffle_by_length' and batch_size > 1:
        # load utterance duration
        train_dur = os.path.join(os.path.dirname(train_label), 'utt2dur')
        assert os.path.isfile(
            train_dur), f"utt2dur file not found: {train_dur}"
        # size of block shuffle
        block_shuffle_size = world_size * batch_size
    else:
        train_dur = None
        block_shuffle_size = 0

    # dataset and dataloader
    train_dataset = Dataset(configs['data_type'],
                            configs['train_data'], #raw.list
                            configs['dataset_args'],
                            lab2id_dict,
                            whole_utt=configs['dataset_args'].get('whole_utt'),
                            reverb_lmdb_file=configs.get('reverb_data', None),
                            noise_lmdb_file=configs.get('noise_data', None),
                            data_dur_file=train_dur,
                            block_shuffle_size=block_shuffle_size,
                            target_dur=configs.get('target_dur', 5.0))
    
    tmp_params_dataloader["num_workers"] = 0
    if tmp_params_dataloader.get("num_workers", 0) == 0:
        tmp_params_dataloader["prefetch_factor"] = None

    train_dataloader = DataLoader(
                            train_dataset,
                            collate_fn=collate_fn,
                            **tmp_params_dataloader)
    if configs['dataset_args'].get('sample_num_per_epoch', 0) > 0:
        sample_num_per_epoch = configs['dataset_args']['sample_num_per_epoch']
    else:
        sample_num_per_epoch = len(train_utt_lab_list)
    epoch_iter = sample_num_per_epoch // world_size // batch_size

    # validation data
    val_dataloader = None
    if configs.get('val_label'):
        val_label = configs['val_label']
        val_utt_lab_list = read_table(val_label)
        total_val = len(val_utt_lab_list)

        if rank == 0:
            logger.info(f"validation data num: {total_val}")

        val_lists = read_lists(configs['val_data'])
        

        if rank == 0:
            logger.info(f"[Validation Subset] Using {len(val_lists)} / {total_val} samples")

        val_dataset = DataList(
            val_lists,
            shuffle=False,           
            repeat_dataset=False,    
            block_shuffle_size=0
        )
        
        val_dataset = Processor(val_dataset, parse_multi_raw)

        val_dataset = Processor(val_dataset, processor.lab_to_id, lab2id_dict)

        val_dataloader = DataLoader(
            val_dataset,
            collate_fn=collate_fn,
            **tmp_params_dataloader
        )

        val_iter = len(val_lists) // world_size // batch_size


    if rank == 0:
        logger.info("<== Dataloaders ==>")
        logger.info("train dataloaders created")
        logger.info('Train epoch iteration number: {}'.format(epoch_iter))
        if val_dataloader is not None:
            logger.info('validation dataloaders created')
            logger.info('Validation iteration number: {}'.format(val_iter))
    
    # model: frontend (optional) => speaker model => projection layer
    logger.info("<== Model ==>")
    frontend_type = configs['dataset_args'].get('frontend', 'fbank')
    if frontend_type != "fbank" and not frontend_type.startswith('lfcc'):
        frontend_args = frontend_type + "_args"
        frontend = frontend_class_dict[frontend_type](
            **configs['dataset_args'][frontend_args],
            sample_rate=configs['dataset_args']['resample_rate'])
        if 'feat_dim' not in configs['model_args'] or configs['model_args']['feat_dim'] < 0:
            configs['model_args']['feat_dim'] = frontend.output_size()
        model = get_model(configs['model'])(**configs['model_args'])
        model.add_module("frontend", frontend)
    else:
        model = get_model(configs['model'])(**configs['model_args'])

    if rank == 0:
        num_params = sum(param.numel() for param in model.parameters())
        logger.info('speaker_model size: {}'.format(num_params))
    if configs['model_init'] is not None:
        logger.info('Load initial model from {}'.format(configs['model_init']))
        load_checkpoint_new(model, configs['model_init'])
    elif checkpoint is None:
        logger.info('Train model from scratch ...')
    # projection layer
    fusion_type = configs.get('fusion_type', 'xattn_add')
    N_refs = configs.get('num_ref_utts', 1)
    base_dim = configs['model_args']['embed_dim']
    if (configs['model_args']['embed_dim'] < 0):  # TODO check
        # #if emb_dim <0, we will reduce dim by emb_dim. like -2 will be dim/2
        if 'multireso' in configs[
                'model'] and configs['model_args']['num_scale'] > 0:
            configs['projection_args']['embed_dim'] = int(
                configs['model_args']['feat_dim'] /
                pow(abs(configs['model_args']['embed_dim']),
                    configs['model_args']['num_scale']))
        else:
            configs['projection_args']['embed_dim'] = int(
                configs['model_args']['feat_dim'] /
                abs(configs['model_args']['embed_dim']))
    else:
        configs['projection_args']['embed_dim'] = base_dim
    configs['projection_args']['num_class'] = len(lab2id_dict)
    configs['projection_args']['do_lm'] = configs.get('do_lm', False)
    if configs['data_type'] != 'feat' and configs['dataset_args'][
            'speed_perturb']:
        # diff speed is regarded as diff lab
        configs['projection_args']['num_class'] *= 3
        if configs.get('do_lm', False):
            logger.info(
                'No speed perturb while doing large margin fine-tuning')
            configs['dataset_args']['speed_perturb'] = False
    projection = get_projection(configs['projection_args'])
    model.add_module("projection", projection)
    if rank == 0:
        # print model
        for line in pformat(model).split('\n'):
            logger.info(line)
        # !!!IMPORTANT!!!
        # Try to export the model by script, if fails, we should refine
        # the code to satisfy the script export requirements
        if frontend_type == 'fbank':
            script_model = torch.jit.script(model)
            script_model.save(os.path.join(model_dir, 'init.zip'))
    
    # ---- fusion module (must be part of model BEFORE DDP) ----
    if fusion_type == 'xattn_add':
        D = configs['model_args']['embed_dim'] 
        fusion_module = MHACrossAttnAdd(
            embed_dim=D,
            num_heads=configs.get('xattn_heads', 4),
            dropout=configs.get('xattn_dropout', 0.1)
        ).to(device)

        model.add_module("fusion_module", fusion_module)
    # If specify checkpoint, load some info from checkpoint.
    # For checkpoint, frontend, speaker model, and projection layer
    # are all needed !!!
    if checkpoint is not None:
        load_checkpoint_new(model, checkpoint)
        start_epoch = int(re.findall(r"(?<=model_)\d*(?=.pt)",
                                     checkpoint)[0]) + 1
        logger.info('Load checkpoint: {}'.format(checkpoint))
    else:
        start_epoch = 1
    logger.info('start_epoch: {}'.format(start_epoch))

    # freeze some pretraining-specific parameters
    for name, param in model.named_parameters():
        if any(k in name for k in ["quantizer", "project_q", "final_proj"]):
            param.requires_grad = False

    # ddp_model
    model.to(device)
    ddp_model = torch.nn.parallel.DistributedDataParallel( 
        model,
        device_ids=[gpu],
        output_device=gpu, 
        find_unused_parameters=True)
    if hasattr(ddp_model, "_set_static_graph"):
        ddp_model._set_static_graph()

    criterion = getattr(torch.nn, configs['loss'])(**configs['loss_args'])
    if rank == 0:
        logger.info("<== Loss ==>")
        logger.info("loss criterion is: " + configs['loss'])

    configs['optimizer_args']['lr'] = configs['scheduler_args']['initial_lr']
    optimizer = getattr(torch.optim,
                        configs['optimizer'])(ddp_model.parameters(),
                                              **configs['optimizer_args'])
    if rank == 0:
        logger.info("<== Optimizer ==>")
        logger.info("optimizer is: " + configs['optimizer'])

    # scheduler
    # Max epochs
    configs['scheduler_args']['num_epochs'] = configs['num_epochs']
    configs['scheduler_args']['epoch_iter'] = epoch_iter
    # here, we consider the batch_size 64 as the base, the learning rate will be
    # adjusted according to the batchsize and world_size used in different setup
    configs['scheduler_args']['scale_ratio'] = 1.0 * world_size * configs[
        'dataloader_args']['batch_size'] / 64
    scheduler = getattr(schedulers,
                        configs['scheduler'])(optimizer,
                                              **configs['scheduler_args'])
    if rank == 0:
        logger.info("<== Scheduler ==>")
        logger.info("scheduler is: " + configs['scheduler'])

    # margin scheduler
    configs['margin_update']['epoch_iter'] = epoch_iter
    margin_scheduler = getattr(schedulers, configs['margin_scheduler'])(
        model=ddp_model.module, **configs['margin_update'])
    if rank == 0:
        logger.info("<== MarginScheduler ==>")

    # save config.yaml
    if rank == 0:
        saved_config_path = os.path.join(configs['exp_dir'], 'config.yaml')
        with open(saved_config_path, 'w') as fout:
            data = yaml.dump(configs)
            fout.write(data)

    if configs.get("save_original_model", False):
        if rank == 0:
            logger.info("Saving the original model.")
            save_checkpoint(model, os.path.join(model_dir, "model_0.pt"))
        dist.barrier(device_ids=[gpu])
        dist.destroy_process_group()
        return
    
    # training
    # dist.barrier(device_ids=[gpu])  # synchronize here
    if rank == 0:
        logger.info("<========== Training process ==========>")
        header = ['Epoch', 'Batch', 'Lr', 'Margin', 'Loss', "Acc"]
        for line in tp.header(header, width=10, style='grid').split('\n'):
            logger.info(line)
    # dist.barrier(device_ids=[gpu])  # synchronize here
    
    scaler = torch.amp.GradScaler('cuda', enabled=configs['enable_amp'])
    # scaler = torch.cuda.amp.GradScaler(enabled=configs['enable_amp'])
    best_val_acc = float('-inf')
    val_no_improvement_count = 0
    early_stop_patience = configs.get('early_stop_patience', 5)
    val_interval = configs.get('validate_interval', 1)

    for epoch in range(start_epoch, configs['num_epochs'] + 1):
        train_dataset.set_epoch(epoch)

        train_epoch(train_dataloader,
                    epoch_iter,
                    ddp_model,
                    criterion,
                    optimizer,
                    scheduler,
                    margin_scheduler,
                    epoch,
                    logger,
                    scaler,
                    device,
                    configs,
                    wandb_log=wandb_run)

        if val_dataloader is not None and (epoch - 1) % val_interval == 0:
            val_loss, val_acc = val_epoch(val_dataloader,
                                          val_iter,
                                          ddp_model,
                                          criterion,
                                          device,
                                          configs,
                                          wandb_log=wandb_run)
            if rank == 0:
                logger.info(
                    f"Validation - Epoch: {epoch}, Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
                )

                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    val_no_improvement_count = 0
                    logger.info(
                        f"New best validation accuracy: {best_val_acc:.6f}")
                    save_checkpoint(model,
                                    os.path.join(model_dir, 'best_model.pt'))
                else:
                    val_no_improvement_count += 1
                    logger.info(
                        f"No improvement for {val_no_improvement_count} validation checks"
                    )
                    if val_no_improvement_count >= early_stop_patience:
                        logger.info(
                            f"Early stopping triggered after {epoch} epochs")
                        save_checkpoint(model,
                                        os.path.join(model_dir, 'final.pt'))
                        break

        if rank == 0:
            if epoch % configs['save_epoch_interval'] == 0 or epoch > configs[
                    'num_epochs'] - configs['num_avg']:
                save_checkpoint(
                    model, os.path.join(model_dir,
                                        'model_{}.pt'.format(epoch)))

    if dist.is_initialized():
        dist.destroy_process_group()
    wandb_run.finish()


if __name__ == '__main__':
    fire.Fire(train)
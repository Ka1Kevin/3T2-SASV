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

import tableprint as tp
import torch
import torchnet as tnt
from torch.nn.utils.rnn import pad_sequence

from wedefense.dataset.dataset_utils import apply_cmvn, spec_aug


def train_epoch(dataloader,
                epoch_iter,
                model,
                criterion,
                optimizer,
                scheduler,
                margin_scheduler,
                epoch,
                logger,
                scaler,
                device,
                configs,
                wandb_log=None):
    """Train the model for one epoch.

    Args:
        dataloader: Training dataloader
        epoch_iter: Number of iterations per epoch
        model: Model to train
        criterion: Loss criterion
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        margin_scheduler: Margin scheduler
        epoch: Current epoch number
        logger: Logger
        scaler: Gradient scaler for mixed precision training
        device: Device to run training on
        configs: Configuration dictionary

    Returns:
        tuple: (training loss, training accuracy, last batch index)
    """
    model.train()
    # By default use average pooling
    loss_meter = tnt.meter.AverageValueMeter()
    acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)

    frontend_type = configs['dataset_args'].get('frontend', 'fbank')
    for i, batch in enumerate(dataloader):
        cur_iter = (epoch - 1) * epoch_iter + i
        scheduler.step(cur_iter)
        margin_scheduler.step(cur_iter)

        targets = batch['labels']
        targets = targets.long().to(device)  # (128)

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
                ref_concat = torch.cat(ref_feat_list, dim=1)  # (768, 250, 13)
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
            concat_feat = torch.cat([ref_concat, test_feat], dim=1)  # (1536, 250, 13)
            concat_feats.append(concat_feat)
        features = pad_sequence(concat_feats, batch_first=True)  # (128, 1536, 250, 13)


        with torch.cuda.amp.autocast(enabled=configs['enable_amp']):
            # apply cmvn
            if configs['dataset_args'].get('cmvn', True):
                features = apply_cmvn(
                    features, **configs['dataset_args'].get('cmvn_args', {}))
            # spec augmentation
            if configs['dataset_args'].get('spec_aug', False):
                features = spec_aug(features,
                                    **configs['dataset_args']['spec_aug_args'])

            outputs = model(features)  # (embed_a,embed_b) in most cases (128, 256)
            embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
            outputs = model.module.projection(embeds, targets)
            # outputs = model.projection(embeds, targets) # (128, 3)
            if isinstance(outputs, tuple):
                outputs, loss = outputs
            else:
                loss = criterion(outputs, targets)

        # loss, acc
        loss_meter.add(loss.item())
        acc_meter.add(outputs.cpu().detach().numpy(), targets.cpu().numpy())

        # update the model
        optimizer.zero_grad()
        # scaler does nothing here if enable_amp=False
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if (wandb_log):
            wandb_log.log({
                "learning_rate": scheduler.get_lr(),
                "train/loss": loss_meter.value()[0],
                "train/acc": acc_meter.value()[0]
            })
        # log
        if (i + 1) % configs['log_batch_interval'] == 0:
            logger.info(
                tp.row((epoch, i + 1, scheduler.get_lr(),
                        margin_scheduler.get_margin()) +
                       (loss_meter.value()[0], acc_meter.value()[0]),
                       width=10,
                       style='grid'))

        if (i + 1) == epoch_iter:
            break

    # Final log for this epoch
    logger.info(
        tp.row(
            (epoch, i + 1, scheduler.get_lr(), margin_scheduler.get_margin()) +
            (loss_meter.value()[0], acc_meter.value()[0]),
            width=10,
            style='grid'))
    


def val_epoch(val_dataloader,
              val_iter,
              model,
              criterion,
              device,
              configs,
              wandb_log=None):
    """Validate the model on the validation set.

    Args:
        val_dataloader: Validation dataloader
        model: Model to validate
        criterion: Loss criterion
        device: Device to run validation on
        configs: Configuration dictionary

    Returns:
        tuple: (validation loss, validation accuracy)
    """
    model.eval()
    val_loss_meter = tnt.meter.AverageValueMeter()
    val_acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)

    frontend_type = configs['dataset_args'].get('frontend', 'fbank')
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            targets = batch['labels'].long().to(device)
            B = targets.shape[0]

            concat_feats = []

            # ---------- 和 train 完全一致的拼接逻辑 ----------
            for b in range(B):

                # ============== 处理 ref ==============
                ref_feat_list = []

                if frontend_type == 'fbank' or frontend_type.startswith('lfcc'):
                    # batch['ref_feats'][b] = (N_ref, T, F)
                    for j in range(batch['ref_feats'][b].shape[0]):
                        ref_feat_list.append(
                            batch['ref_feats'][b][j].float().to(device)
                        )
                else:
                    # s3prl 前端
                    for j in range(batch['refs'][b].shape[0]):
                        wav = batch['refs'][b][j].unsqueeze(0).float().to(device)
                        wav_len = torch.LongTensor([wav.shape[1]]).to(device)
                        with torch.cuda.amp.autocast(enabled=configs['enable_amp']):
                            feat, _ = model.module.frontend(wav, wav_len)
                        ref_feat_list.append(feat.squeeze(0))

                if len(ref_feat_list) > 0:
                    ref_concat = torch.cat(ref_feat_list, dim=1)
                else:
                    ref_concat = torch.empty(0).to(device)

                # ============== 处理 test ==============
                if frontend_type == 'fbank' or frontend_type.startswith('lfcc'):
                    test_feat = batch['feat'][b].float().to(device)
                else:
                    test_wav = batch['test'][b].unsqueeze(0).float().to(device)
                    wav_len = torch.LongTensor([test_wav.shape[1]]).to(device)
                    with torch.cuda.amp.autocast(enabled=configs['enable_amp']):
                        test_feat, _ = model.module.frontend(test_wav, wav_len)
                    test_feat = test_feat.squeeze(0)

                # ============== 拼接 ref + test ==============
                concat_feat = torch.cat([ref_concat, test_feat], dim=1)
                concat_feats.append(concat_feat)

            # pad 到同一长度
            features = pad_sequence(concat_feats, batch_first=True).to(device)

            # ---------- CMVN、SpecAug ----------
            with torch.cuda.amp.autocast(enabled=configs['enable_amp']):
                if configs['dataset_args'].get('cmvn', True):
                    features = apply_cmvn(
                        features,
                        **configs['dataset_args'].get('cmvn_args', {})
                    )

                # (val 时一般不做 spec_aug，这里保持 train 行为一致即可)
                if configs['dataset_args'].get('spec_aug', False):
                    features = spec_aug(
                        features,
                        **configs['dataset_args']['spec_aug_args']
                    )

                # Forward
                outputs = model(features)
                embeds = outputs[-1] if isinstance(outputs, tuple) else outputs

                outputs = model.module.projection(embeds, targets)

                if isinstance(outputs, tuple):
                    outputs, loss = outputs
                else:
                    loss = criterion(outputs, targets)

            # ---------- 统计 ----------
            val_loss_meter.add(loss.item())
            val_acc_meter.add(outputs.cpu().numpy(), targets.cpu().numpy())

            if wandb_log:
                wandb_log.log({
                    "val/loss": val_loss_meter.value()[0],
                    "val/acc": val_acc_meter.value()[0]
                })

            if (i + 1) == val_iter:
                break

    return val_loss_meter.value()[0], val_acc_meter.value()[0]

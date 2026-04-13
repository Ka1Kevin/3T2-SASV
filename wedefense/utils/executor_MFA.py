import tableprint as tp
import torch
import torchnet as tnt

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
    """Train the model for one epoch (MFA_Conformer only)."""
    model.train()

    torch.autograd.set_detect_anomaly(True)

    loss_meter = tnt.meter.AverageValueMeter()
    acc_meter = tnt.meter.AverageValueMeter()
    # acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
    
    for i, batch in enumerate(dataloader):
        cur_iter = (epoch - 1) * epoch_iter + i
        scheduler.step(cur_iter)
        margin_scheduler.step(cur_iter)

        test = batch['test'].to(device).float()  # (B, T) (200, 80000)
        refs = batch['refs'].to(device).float()   # (B, N_refs, T) (200, 1, 80000)
        targets = batch['labels'].long().to(device)
        
        # with torch.cuda.amp.autocast(enabled=configs['enable_amp']):
        with torch.amp.autocast('cuda', enabled=configs['enable_amp']):
            # test_embeds = model(test) # (200, 192)
            B, N_refs, T = refs.shape
            all_wavs = torch.cat([test.unsqueeze(1), refs], dim=1)   # (B, 1+N_refs, T)
            all_wavs = all_wavs.reshape(B * (1 + N_refs), T)         # (B*(1+N_refs), T)
            all_embeds = model(all_wavs)                             # (B*(1+N_refs), D)
            all_embeds = all_embeds.reshape(B, 1 + N_refs, -1)       # (B, 1+N_refs, D)

            embeds = all_embeds.reshape(B, -1)

            # outputs = model.module.projection(embeds, targets)
            loss, prec1 = criterion(embeds, targets)
            
        loss_meter.add(float(loss.detach().cpu().item()))
        acc_meter.add(prec1.item())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if wandb_log:
            wandb_log.log({
                "learning_rate": scheduler.get_lr(),
                "train/loss": loss_meter.value()[0],
                "train/acc": acc_meter.value()[0]
            })

        if (i + 1) % configs['log_batch_interval'] == 0:
            logger.info(
                tp.row((epoch, i + 1, scheduler.get_lr(),
                        margin_scheduler.get_margin(),
                        loss_meter.value()[0], acc_meter.value()[0]),
                       width=10, style='grid'))

        if (i + 1) == epoch_iter:
            break

    logger.info(
        tp.row(
            (epoch, i + 1, scheduler.get_lr(), margin_scheduler.get_margin(),
             loss_meter.value()[0], acc_meter.value()[0]),
            width=10, style='grid'))


def val_epoch(val_dataloader,
              val_iter,
              model,
              criterion,
              device,
              configs,
              wandb_log=None):
    """Validate the model on the validation set (MFA_Conformer only)."""
    model.eval()
    val_loss_meter = tnt.meter.AverageValueMeter()
    val_acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)

    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            wavs = batch['wav'].to(device).float()
            targets = batch['label'].long().to(device)

            with torch.cuda.amp.autocast(enabled=configs['enable_amp']):
                embeds = model(wavs)
                outputs = model.module.projection(embeds, targets)
                if isinstance(outputs, tuple):
                    outputs, loss = outputs
                else:
                    loss = criterion(outputs, targets)

            val_loss_meter.add(loss.item())
            val_acc_meter.add(outputs.cpu().detach().numpy(), targets.cpu().numpy())

            if wandb_log:
                wandb_log.log({
                    "val/loss": val_loss_meter.value()[0],
                    "val/acc": val_acc_meter.value()[0]
                })

            if (i + 1) == val_iter:
                break

    return val_loss_meter.value()[0], val_acc_meter.value()[0]

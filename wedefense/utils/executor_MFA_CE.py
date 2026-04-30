import tableprint as tp
import torch
import torchnet as tnt
import torch.nn as nn
class MHACrossAttnAdd(nn.Module):
    def __init__(self, embed_dim=192, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(embed_dim)
        self.emb_dim = embed_dim
        self.num_heads = num_heads

    def forward(self, test_embed, refs_embed):
        # test_embed: (B,D), refs_embed: (B,N,D)
        q = test_embed.unsqueeze(1)          # (B,1,D)
        k = refs_embed
        v = refs_embed
        ctx, _ = self.attn(q, k, v)  # (B,1,D)
        ctx = ctx.squeeze(1)                # (B,D)
        return self.ln(test_embed + ctx)    # (B,D)
    
def build_embeds(model, test, refs, configs):
    """
    test: (B,T)
    refs: (B,N_refs,T)
    return embeds for projection
    """
    fusion_type = configs.get('fusion_type', 'xattn_add')
    B, N_refs, T = refs.shape

    if fusion_type == 'feat_cat':
        all_wavs = torch.cat([test.unsqueeze(1), refs], dim=1)   # (B,1+N,T)
        all_wavs = all_wavs.reshape(B * (1 + N_refs), T)         # (B*(1+N),T)
        all_embeds = model(all_wavs)                             # (B*(1+N),D)
        all_embeds = all_embeds.reshape(B, 1 + N_refs, -1)       # (B,1+N,D)
        embeds = all_embeds.reshape(B, -1)                       # (B,(1+N)*D)
        return embeds

    elif fusion_type == 'xattn_add':
        if not hasattr(model.module, "fusion_module"):
            raise RuntimeError(
                "fusion_type='xattn_add' but model.module.fusion_module is missing. "
                "Make sure you model.add_module('fusion_module', ...) BEFORE DDP."
            )
        fusion = model.module.fusion_module
        
        refs_flat = refs.reshape(B * N_refs, T)                 # (B*N, T)
        all_wav = torch.cat([test, refs_flat], dim=0)           # (B + B*N, T)
        all_emb = model(all_wav)                                # (B + B*N, D)
        test_embed = all_emb[:B]                                # (B, D)
        refs_embed = all_emb[B:].reshape(B, N_refs, -1)         # (B, N, D)

        embeds = fusion(test_embed, refs_embed)   # (B,D)
        return embeds
    
    elif fusion_type == 'time_cat':
        # time concat at waveform level, forward once
        refs_time = refs.reshape(B, N_refs * T)          # (B, N*T)
        wav_cat = torch.cat([test, refs_time], dim=1)    # (B, (N+1)*T)
        embeds = model(wav_cat)                          # (B, D)
        return embeds

    else:
        raise ValueError(f"Unknown fusion_type: {fusion_type}")


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
    acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
    
    for i, batch in enumerate(dataloader):
        model.module._debug_step = i
        cur_iter = (epoch - 1) * epoch_iter + i
        scheduler.step(cur_iter)
        with torch.no_grad():
            margin_scheduler.step(cur_iter)

        test = batch['test'].to(device).float()  # (B, T) (200, 80000)
        refs = batch['refs'].to(device).float()   # (B, N_refs, T) (200, 1, 80000)
        targets = batch['labels'].long().to(device)
        
        with torch.amp.autocast('cuda', enabled=configs['enable_amp']):
            embeds = build_embeds(model, test, refs, configs)
            logits = model.module.projection(embeds, targets)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = criterion(logits, targets)

        loss_meter.add(loss.item())
        acc_meter.add(logits.cpu().detach().numpy(), targets.cpu().numpy())

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
    model.eval()
    val_loss_meter = tnt.meter.AverageValueMeter()
    val_acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)

    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            model.module._debug_step = i
            test = batch['test'].to(device).float()    # (B, T)
            refs = batch['refs'].to(device).float()    # (B, N_refs, T)
            targets = batch['labels'].long().to(device)

            with torch.amp.autocast('cuda', enabled=configs['enable_amp']):
                embeds = build_embeds(model, test, refs, configs)
                expected = configs['projection_args']['embed_dim']
                logits = model.module.projection(embeds, targets)
                if isinstance(logits, tuple):
                    logits = logits[0]
                loss = criterion(logits, targets)

            val_loss_meter.add(loss.item())
            val_acc_meter.add(logits.cpu().detach().numpy(), targets.cpu().numpy())

            if wandb_log:
                wandb_log.log({
                    "val/loss": val_loss_meter.value()[0],
                    "val/acc": val_acc_meter.value()[0]
                })

            if (i + 1) == val_iter:
                break

    return val_loss_meter.value()[0], val_acc_meter.value()[0]
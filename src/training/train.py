import json
import math
import os
import time

import torch
from open_clip.loss import ClipLoss
from torch.utils.data import DataLoader

from src.data.dataset import RadiologyNETDataset
from src.data.sampler import ModalityBatchSampler
from src.evaluation.validate import recall_score
from src.models.biomedclip import load_biomedclip, set_freeze_mode
from src.training.losses import MaskedClipLoss
from src.training.scheduler import cosine_lr_with_warmup

clip_loss = ClipLoss()
masked_clip_loss = MaskedClipLoss()


def clamp_logit_scale(model):
    with torch.no_grad():
        model.logit_scale.clamp_(0, math.log(100))


def build_param_groups(model, base_lr, weight_decay, head_lr_mult=1.0):
    # two things going on here:
    # 1. open_clip's own training script skips weight decay on 1d params (norms, biases,
    #    logit_scale) - decaying those regularises nothing and just fights the model
    # 2. the projection heads are what most need to move for a new domain, the pretrained
    #    blocks should shift more gently, hence head_lr_mult
    def is_no_decay(name, p):
        return p.ndim < 2 or 'bias' in name or 'logit_scale' in name

    buckets = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'logit_scale' in name:
            # the learned temperature stays on the base lr, open_clip never gives it a
            # multiplier and a 10x on one scalar is an easy way to destabilise the loss
            where = 'logit_scale'
        elif 'visual.trunk' in name or 'text.transformer' in name:
            where = 'backbone'
        else:
            where = 'head'
        buckets.setdefault((where, is_no_decay(name, p)), []).append(p)

    groups = []
    for (where, no_decay), params in sorted(buckets.items()):
        groups.append({
            'params': params,
            'lr': base_lr * head_lr_mult if where == 'head' else base_lr,
            'weight_decay': 0.0 if no_decay else weight_decay,
            'group_name': f'{where}_{"nodecay" if no_decay else "decay"}',
        })
    return groups


def train_one_epoch(model, loader, optimizer, lr_adjuster, step, amp_dtype, grad_clip_norm=None, max_steps=None, log_every=25):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        if max_steps is not None and n_batches >= max_steps:
            break
        t0 = time.time()
        lr_adjuster(step)

        if len(batch) == 3:
            images, texts, text_group = batch
            text_group = text_group.cuda()
        else:
            images, texts = batch
            text_group = None
        images, texts = images.cuda(), texts.cuda()
        optimizer.zero_grad()

        with torch.autocast(device_type='cuda', dtype=amp_dtype):
            image_features = model.encode_image(images, normalize=True)
            text_features = model.encode_text(texts, normalize=True)
            if text_group is not None:
                loss = masked_clip_loss(image_features, text_features,
                                        model.logit_scale.exp(), text_group)
            else:
                loss = clip_loss(image_features, text_features, model.logit_scale.exp())
        loss.backward()

        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], grad_clip_norm
            )
        optimizer.step()
        clamp_logit_scale(model)

        total_loss += loss.item()
        n_batches += 1
        step += 1

        if n_batches % log_every == 0:
            gb = torch.cuda.max_memory_allocated() / 1e9
            print(f"  step {n_batches}  loss={loss.item():.4f}  {time.time()-t0:.2f}s/step  peak_vram={gb:.2f}GB", flush=True)

    return total_loss / n_batches, step


@torch.no_grad()
def evaluate(model, loader, amp_dtype, collect_embeddings=False):
    # collecting embeddings here rather than in a second pass - val_loss and val recall
    # need the exact same forward pass, no point paying for it twice
    model.eval()
    total_loss = 0.0
    n_batches = 0
    image_embeds, text_embeds = [], []
    for images, texts in loader:
        images, texts = images.cuda(), texts.cuda()
        with torch.autocast(device_type='cuda', dtype=amp_dtype):
            image_features = model.encode_image(images, normalize=True)
            text_features = model.encode_text(texts, normalize=True)
            loss = clip_loss(image_features, text_features, model.logit_scale.exp())
        total_loss += loss.item()
        n_batches += 1
        if collect_embeddings:
            image_embeds.append(image_features.float().cpu())
            text_embeds.append(text_features.float().cpu())

    mean_loss = total_loss / n_batches
    if collect_embeddings:
        return mean_loss, torch.cat(image_embeds), torch.cat(text_embeds)
    return mean_loss


def train(
    data_root='data', images_root='data/images',
    batch_size=96, epochs=20, base_lr=1e-5, weight_decay=0.1,
    warmup_steps=150, grad_clip_norm=1.0,
    amp_dtype=torch.bfloat16, freeze_mode='partial', n_unfrozen_blocks=2,
    head_lr_mult=1.0, grad_checkpointing=False, num_workers=4,
    checkpoint_path='checkpoints/best.pt', patience=5,
    max_steps=None, select_on='recall', history_path=None,
    mask_false_negatives=False, hard_negatives=None,
):
    model, preprocess_train, preprocess_val, tokenizer = load_biomedclip()
    set_freeze_mode(model, freeze_mode, n_unfrozen_blocks)
    if grad_checkpointing:
        model.set_grad_checkpointing(True)
    model = model.cuda()

    train_ds = RadiologyNETDataset(split='train', data_root=data_root, images_root=images_root,
                                    tokenizer=tokenizer, image_transform=preprocess_train,
                                    return_text_group=mask_false_negatives)
    val_ds = RadiologyNETDataset(split='val', data_root=data_root, images_root=images_root,
                                  tokenizer=tokenizer, image_transform=preprocess_val,
                                  random_slice=False)
    val_metadata = val_ds.rows.reset_index().rename(columns={'index': 'id'})

    train_kwargs = dict(num_workers=num_workers)
    if num_workers > 0:
        train_kwargs.update(persistent_workers=True, prefetch_factor=2)

    batch_sampler = None
    if hard_negatives is not None:
        batch_sampler = ModalityBatchSampler(
            train_ds.rows.Modality.values, batch_size,
            random_fraction=1.0 - hard_negatives,
        )
        print(f'hard negatives on: {hard_negatives:.0%} of rows in modality-grouped '
              f'batches, {len(batch_sampler)} batches/epoch', flush=True)
        train_loader = DataLoader(train_ds, batch_sampler=batch_sampler, **train_kwargs)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **train_kwargs)

    # val stays on 0 workers on purpose. it's only ~46 batches once an epoch, but with
    # persistent workers it kept 4 extra processes alive for the whole run, each holding
    # its own ~800MB torch import. that pushed the windows paging file to 10GB and filled
    # the disk mid-run, which is what killed v2 the first time round
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    param_groups = build_param_groups(model, base_lr, weight_decay, head_lr_mult)
    for g in param_groups:
        n = sum(p.numel() for p in g['params'])
        print(f"param group {g['group_name']}: {n:,} params, lr={g['lr']:.2e}, wd={g['weight_decay']}", flush=True)
    optimizer = torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=weight_decay)

    steps_per_epoch = max_steps if max_steps is not None else len(train_loader)
    total_steps = steps_per_epoch * epochs
    lr_adjuster = cosine_lr_with_warmup(optimizer, min(warmup_steps, total_steps // 2), total_steps)

    os.makedirs(os.path.dirname(checkpoint_path) or '.', exist_ok=True)

    best_score = None
    epochs_without_improvement = 0
    step = 0
    history = []

    for epoch in range(epochs):
        t0 = time.time()
        if batch_sampler is not None:
            batch_sampler.set_epoch(epoch)
        train_loss, step = train_one_epoch(model, train_loader, optimizer, lr_adjuster, step, amp_dtype, grad_clip_norm, max_steps)

        if select_on == 'recall':
            val_loss, img_e, txt_e = evaluate(model, val_loader, amp_dtype, collect_embeddings=True)
            score, t2i, i2t = recall_score(img_e, txt_e, val_metadata)
            current = score
        else:
            val_loss = evaluate(model, val_loader, amp_dtype)
            score, t2i, i2t = None, None, None
            current = -val_loss

        row = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
               'logit_scale': model.logit_scale.item(), 'epoch_seconds': time.time() - t0}
        if score is not None:
            row.update({'val_recall_score': score,
                        'val_t2i_r10': t2i[10], 'val_i2t_r10': i2t[10]})
        history.append(row)

        msg = f"epoch {epoch}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
        if score is not None:
            msg += f"  val_R@10={score:.4f} (t2i {t2i[10]:.3f} / i2t {i2t[10]:.3f})"
        msg += f"  logit_scale={model.logit_scale.item():.4f}  {row['epoch_seconds']:.0f}s"
        print(msg, flush=True)

        if best_score is None or current > best_score:
            best_score = current
            epochs_without_improvement = 0
            # only the trainable tensors - the frozen ones are still exactly the pretrained
            # weights, so re-saving them per variant just burns disk (783MB vs ~120MB)
            trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
            partial_state = {k: v for k, v in model.state_dict().items() if k in trainable_names}
            torch.save({'model_state_dict': partial_state, 'partial_state': True,
                        'freeze_mode': freeze_mode, 'n_unfrozen_blocks': n_unfrozen_blocks,
                        'epoch': epoch, 'val_loss': val_loss, 'val_recall_score': score},
                       checkpoint_path)
            print(f"  saved checkpoint (epoch {epoch})", flush=True)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"no improvement for {patience} epochs, stopping early", flush=True)
                break

    if history_path:
        os.makedirs(os.path.dirname(history_path) or '.', exist_ok=True)
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

    return history


if __name__ == '__main__':
    train()

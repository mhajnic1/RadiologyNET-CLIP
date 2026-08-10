import json
import os
import traceback

import torch

from src.training.train import train

# every variant keeps its own checkpoint so checkpoints/best.pt (what phase 7 and 8 report)
# is never overwritten while I'm experimenting
CHECKPOINT_DIR = 'checkpoints/tuning'
RESULTS_DIR = 'results/tuning'

# batch 64 not 96: the desktop is holding ~1.2GB of the 6.44GB card and 96 OOMs now,
# phase 5/6's headroom numbers were measured with less open. epochs 10/patience 3 because
# phase 6's useful improvement was over by epoch 3-4 anyway
BASE = dict(
    batch_size=64, epochs=8, base_lr=1e-5, weight_decay=0.1,
    warmup_steps=150, grad_clip_norm=1.0, freeze_mode='partial',
    n_unfrozen_blocks=2, head_lr_mult=1.0, grad_checkpointing=False,
    num_workers=4, patience=3, select_on='recall',
)

VARIANTS = {
    # phase 6's settings again, but with the fixes (no decay on 1d params, deterministic
    # eval slices, recall-based selection) so everything after this is a fair comparison
    'v1_baseline': {},

    # heads carry the domain shift, the pretrained blocks shouldn't move as fast
    'v2_head_lr': dict(head_lr_mult=10.0),

    # phase 6 overfit hard past epoch 3, so cut trainable capacity and decay harder
    'v3_regularised': dict(head_lr_mult=10.0, weight_decay=0.5, n_unfrozen_blocks=1),

    # contrastive learning lives on in-batch negatives, so more of them per step should
    # be a harder and more useful task. only way to fit it here is grad checkpointing,
    # which costs ~1.8x per step, so this one is the expensive experiment.
    # batch 192 was the original plan and it does not work: 5.84GB against ~5GB usable,
    # so windows spills to system RAM instead of raising OOM and the step time goes from
    # ~4s to 494s. measured 128 first this time, 4.25GB and 2.82s/step, which is clean
    'v4_big_batch': dict(head_lr_mult=10.0, grad_checkpointing=True, batch_size=128,
                         warmup_steps=75),

    # same-exam images share byte-identical diagnosis text, so ClipLoss is pushing apart
    # pairs that are actually correct in ~52% of batches. mask them out of the softmax.
    # compare against v1, same config otherwise
    'v5_masked': dict(mask_false_negatives=True),

    # 98.8% of top-10 retrievals already share the query's modality, yet ~49 of 63
    # negatives in a random batch are cross-modality, so most of the contrastive signal
    # is spent on a problem the model already solved. group batches by modality to make
    # the negatives actually hard. needs the mask, because grouping concentrates
    # same-exam images and takes false negatives from 0.52 to ~3-4 per batch
    'v6_hardneg': dict(mask_false_negatives=True, hard_negatives=0.7),
}


def run(names=None, resume=True):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    names = names or list(VARIANTS)
    summary_path = os.path.join(RESULTS_DIR, 'summary.json')

    # start from whatever already finished, so a crashed run can be picked up without
    # throwing away variants that completed (learned this the hard way)
    summary = {}
    if resume and os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        names = [n for n in names if summary.get(n, {}).get('status') != 'ok']
        print(f'resuming, already done: {[k for k in summary if summary[k].get("status") == "ok"]}', flush=True)
    print(f'running: {names}', flush=True)

    for name in names:
        cfg = dict(BASE)
        cfg.update(VARIANTS[name])
        cfg['checkpoint_path'] = os.path.join(CHECKPOINT_DIR, f'{name}.pt')
        cfg['history_path'] = os.path.join(RESULTS_DIR, f'{name}_history.json')

        print(f"\n{'='*70}\n{name}: {VARIANTS[name] or 'base config'}\n{'='*70}", flush=True)
        try:
            history = train(**cfg)
        except Exception:
            traceback.print_exc()
            print(f'{name} FAILED, moving on', flush=True)
            summary[name] = {'status': 'failed'}
            continue
        finally:
            torch.cuda.empty_cache()

        best = max(history, key=lambda r: r.get('val_recall_score', -1))
        summary[name] = {
            'status': 'ok',
            'config': {k: v for k, v in cfg.items() if k not in ('checkpoint_path', 'history_path')},
            'best_epoch': best['epoch'],
            'best_val_recall_score': best.get('val_recall_score'),
            'best_val_t2i_r10': best.get('val_t2i_r10'),
            'best_val_i2t_r10': best.get('val_i2t_r10'),
            'val_loss_at_best': best['val_loss'],
            'epochs_run': len(history),
        }
        print(f"{name} best: epoch {best['epoch']}, "
              f"val_R@10 {best.get('val_recall_score'):.4f}", flush=True)

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

    return summary


if __name__ == '__main__':
    import sys
    run(sys.argv[1:] or None)

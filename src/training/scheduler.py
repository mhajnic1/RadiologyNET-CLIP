import math


def cosine_lr_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    # scale each group off its own starting lr, otherwise a layer-wise lr multiplier
    # gets flattened away the first time the schedule fires
    base_lrs = [group['lr'] for group in optimizer.param_groups]

    def _set_lr(step):
        if step < warmup_steps:
            scale = (step + 1) / warmup_steps
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            scale = 0.5 * (1 + math.cos(math.pi * progress))
        for group, base in zip(optimizer.param_groups, base_lrs):
            group['lr'] = base * scale
        return scale

    return _set_lr

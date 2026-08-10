import os

import torch


def make_soup(checkpoint_paths, out_path):
    # uniform model soup: average the weights of several fine-tunes of the same base
    # model. works because they all started from the same pretrained checkpoint and
    # stay in one loss basin, so the average is a valid model rather than nonsense.
    # Wortsman et al. 2022, arXiv:2203.05482
    states, ingredients = [], []
    for p in checkpoint_paths:
        ck = torch.load(p, map_location='cpu', weights_only=False)
        if not ck.get('partial_state'):
            raise ValueError(f'{p} is a full state dict, soup expects the partial format')
        states.append(ck['model_state_dict'])
        ingredients.append({
            'path': p,
            'epoch': ck.get('epoch'),
            'val_recall_score': ck.get('val_recall_score'),
            'freeze_mode': ck.get('freeze_mode'),
            'n_unfrozen_blocks': ck.get('n_unfrozen_blocks'),
        })

    keys = set(states[0])
    for p, s in zip(checkpoint_paths[1:], states[1:]):
        if set(s) != keys:
            missing = keys ^ set(s)
            raise ValueError(
                f'{p} has a different trainable parameter set, cannot be souped '
                f'({len(missing)} keys differ). variants must share freeze_mode and '
                f'n_unfrozen_blocks'
            )

    souped = {k: torch.stack([s[k].float() for s in states]).mean(0) for k in keys}

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    torch.save({
        'model_state_dict': souped,
        'partial_state': True,
        'freeze_mode': ingredients[0]['freeze_mode'],
        'n_unfrozen_blocks': ingredients[0]['n_unfrozen_blocks'],
        'soup_ingredients': ingredients,
        'epoch': None,
        'val_loss': None,
        'val_recall_score': None,
    }, out_path)

    print(f'souped {len(states)} checkpoints into {out_path}')
    for ing in ingredients:
        print(f"  {ing['path']}  epoch {ing['epoch']}  val {ing['val_recall_score']:.4f}")
    return out_path

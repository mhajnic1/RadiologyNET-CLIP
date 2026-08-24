import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from src.data.dataset import RadiologyNETDataset
from src.data.paths import get_image_dir, list_slices


class MultiViewImageDataset(Dataset):
    """One row per (image id, png file) pair.

    CT ids carry two files that are the same anatomical slice under two different
    windowings, typically lung and mediastinal, and those windows show genuinely
    different things. XA ids carry temporal frames. Evaluation currently keeps only
    the first file and throws the rest away, on 18.6% of test rows. This enumerates
    all of them so their embeddings can be pooled back per image id.

    max_views caps XA, which goes up to 554 frames and would otherwise dominate.
    """

    def __init__(self, rows, images_root, image_transform, max_views=8):
        self.images_root = images_root
        self.image_transform = image_transform

        self.pairs = []
        for row_pos, (image_id, modality) in enumerate(zip(rows.index, rows.Modality)):
            names = list_slices(images_root, image_id, modality)
            if len(names) > max_views:
                # evenly spaced rather than the first n, so an XA sequence is sampled
                # across its whole span instead of just the opening frames
                idx = np.linspace(0, len(names) - 1, max_views).round().astype(int)
                names = [names[i] for i in sorted(set(idx.tolist()))]
            for name in names:
                self.pairs.append((row_pos, image_id, modality, name))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        row_pos, image_id, modality, name = self.pairs[i]
        folder = get_image_dir(self.images_root, image_id, modality)
        image = Image.open(os.path.join(folder, name))
        return self.image_transform(image), row_pos


@torch.no_grad()
def compute_multiview_image_embeddings(split, model, preprocess_val, tokenizer,
                                        data_root='data', images_root='data/images',
                                        batch_size=96, max_views=8):
    base = RadiologyNETDataset(split=split, data_root=data_root, images_root=images_root,
                               tokenizer=tokenizer, image_transform=preprocess_val,
                               random_slice=False)
    ds = MultiViewImageDataset(base.rows, images_root, preprocess_val, max_views=max_views)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    n_rows = len(base.rows)
    summed, counts = None, torch.zeros(n_rows)

    model.eval()
    for images, row_pos in loader:
        images = images.cuda()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            feats = model.encode_image(images, normalize=True)
        feats = feats.float().cpu()
        if summed is None:
            summed = torch.zeros(n_rows, feats.shape[1])
        summed.index_add_(0, row_pos, feats)
        counts.index_add_(0, row_pos, torch.ones(row_pos.shape[0]))

    assert (counts > 0).all(), 'some rows got no views'
    pooled = summed / counts.unsqueeze(1)
    # averaging unit vectors gives something shorter than unit length, and every
    # downstream metric assumes cosine similarity over normalised vectors
    pooled = torch.nn.functional.normalize(pooled, dim=-1)

    metadata = base.rows.reset_index().rename(columns={'index': 'id'})
    return pooled, metadata, len(ds), counts

import numpy as np
from torch.utils.data import Sampler


class ModalityBatchSampler(Sampler):
    """Builds batches from a single modality so the in-batch negatives are actually hard.

    Measured on the phase 9 checkpoint: 98.8% of top-10 retrievals already share the
    query's modality, so the model has effectively solved modality. But in a random
    batch of 64, ~49 of the 63 negatives are cross-modality, meaning most of the
    contrastive signal is spent re-learning something it already knows. Grouping by
    modality makes all 63 negatives require discriminating anatomy and findings.

    random_fraction keeps some ordinary mixed batches so cross-modality separation
    doesn't drift, since the eval pool is mixed.
    """

    def __init__(self, modalities, batch_size, random_fraction=0.3, seed=0, drop_last=True):
        self.modalities = np.asarray(modalities)
        self.batch_size = batch_size
        self.random_fraction = random_fraction
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, epoch):
        # different grouping each epoch, otherwise every epoch sees identical batches
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        n = len(self.modalities)
        order = rng.permutation(n)

        n_random = int(self.random_fraction * n)
        mixed_pool = list(order[:n_random])
        grouped_pool = order[n_random:]

        batches = []
        for m in np.unique(self.modalities):
            sel = grouped_pool[self.modalities[grouped_pool] == m]
            n_full = len(sel) // self.batch_size
            for b in range(n_full):
                batches.append(sel[b * self.batch_size:(b + 1) * self.batch_size].tolist())
            # whatever doesn't fill a batch goes back into the mixed pool rather than
            # forming a tiny batch with far fewer negatives
            mixed_pool.extend(sel[n_full * self.batch_size:].tolist())

        pool = np.array(mixed_pool, dtype=int)
        pool = rng.permutation(pool)
        n_full = len(pool) // self.batch_size
        for b in range(n_full):
            batches.append(pool[b * self.batch_size:(b + 1) * self.batch_size].tolist())
        if not self.drop_last and len(pool) % self.batch_size:
            batches.append(pool[n_full * self.batch_size:].tolist())

        rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return len(self.modalities) // self.batch_size

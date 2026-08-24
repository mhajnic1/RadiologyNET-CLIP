import os
import random

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from src.data.paths import get_image_dir, list_slices


class RadiologyNETDataset(Dataset):
    # random_slice is the phase 4 augmentation decision, but it has to be off for val/test -
    # otherwise the same checkpoint embeds a different picture every run and the numbers
    # aren't reproducible (only bites the ~5% of rows with more than one slice, mostly XA)
    # return_text_group adds an integer id per unique diagnosis string, so the loss can
    # spot two rows in a batch that share identical text and stop treating them as
    # negatives of each other. off by default so the eval code keeps its 2-tuple
    def __init__(self, split, data_root, images_root, tokenizer, image_transform,
                 random_slice=True, return_text_group=False,
                 text_column='DIAGNOSIS_TRUNCATED'):
        splits = pd.read_csv(os.path.join(data_root, 'splits.csv'))
        metainfo = pd.read_csv(os.path.join(data_root, 'metainformation.csv'), index_col='id')

        # split=None means every row, which the demo app needs so it can search the
        # whole dataset. the split label is kept as a column either way so callers
        # can still tell train/val/test apart
        if split is not None:
            exam_ids = set(splits[splits.split == split].ExamID)
            metainfo = metainfo[metainfo.ExamID.isin(exam_ids)]

        # text_column picks which truncation to feed the model, since the original CLIP
        # only has 77 tokens against BiomedCLIP's 256 and needs its own. always exposed
        # downstream as DIAGNOSIS_TRUNCATED so the evaluation code stays unchanged
        diagnoses = pd.read_csv(os.path.join(data_root, 'diagnoses_final.csv'), encoding='utf-8-sig')
        diagnoses = diagnoses.set_index('ExamID')[text_column].rename('DIAGNOSIS_TRUNCATED')

        self.rows = metainfo.join(diagnoses, on='ExamID')
        self.rows['split'] = metainfo.ExamID.map(splits.set_index('ExamID').split)
        self.images_root = images_root
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.random_slice = random_slice
        self.return_text_group = return_text_group
        # factorize gives identical text the same code, which is exactly the grouping
        # the loss mask needs (catches cross-exam duplicates too, not just same-exam)
        self.text_group = pd.factorize(self.rows.DIAGNOSIS_TRUNCATED)[0]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        image_id = self.rows.index[idx]
        row = self.rows.iloc[idx]

        slices = list_slices(self.images_root, image_id, row.Modality)
        chosen = random.choice(slices) if self.random_slice else slices[0]
        folder = get_image_dir(self.images_root, image_id, row.Modality)

        image = Image.open(os.path.join(folder, chosen))
        image = self.image_transform(image)

        text = self.tokenizer([row.DIAGNOSIS_TRUNCATED])[0]

        if self.return_text_group:
            return image, text, int(self.text_group[idx])
        return image, text

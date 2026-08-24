import os

import numpy as np
import torch

from src.data.paths import get_image_dir, list_slices
from src.evaluation.embed import load_trained_model

GALLERY_PATH = 'data/embeddings/gallery_all.pt'


class Gallery:
    """Precomputed embeddings plus the search both directions runs on.

    Everything in the gallery is embedded up front, so a query only costs one text
    encode and one matrix multiply. The image tower is never needed at query time.
    """

    def __init__(self, gallery_path=GALLERY_PATH, images_root='data/images', device=None):
        blob = torch.load(gallery_path, weights_only=False)
        self.image_embeds = blob['image_embeds']
        self.text_embeds = blob['text_embeds']
        self.meta = blob['metadata']
        self.checkpoint = blob['checkpoint']
        self.images_root = images_root

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, _, self.tokenizer = load_trained_model(self.checkpoint)

        self.diag = self.meta['DIAGNOSIS_TRUNCATED'].values
        self.is_test = (self.meta['split'] == 'test').values

        # one entry per unique diagnosis, which is the pool image->text ranks against.
        # duplicate text embeds identically so keeping duplicates would just pad the
        # results with the same string over and over
        seen, keep = set(), []
        for i, d in enumerate(self.diag):
            if d not in seen:
                seen.add(d)
                keep.append(i)
        self.text_pool_idx = np.array(keep)

    # ---------- helpers ----------

    def _scope_mask(self, test_only):
        return self.is_test if test_only else np.ones(len(self.diag), dtype=bool)

    def views(self, row):
        """Every png for this image. CT gives two windowings, XA gives frames."""
        m = self.meta.iloc[row]
        folder = get_image_dir(self.images_root, m.id, m.Modality)
        return [os.path.join(folder, n) for n in list_slices(self.images_root, m.id, m.Modality)]

    def encode_query(self, text):
        tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_text(tokens, normalize=True)
        return feats.float().cpu()[0]

    def sample_rows(self, n, test_only=False, seed=None):
        mask = self._scope_mask(test_only)
        pool = np.flatnonzero(mask)
        rng = np.random.default_rng(seed)
        return rng.choice(pool, size=min(n, len(pool)), replace=False).tolist()

    def test_diagnoses(self, limit=300):
        seen, out = set(), []
        for i in np.flatnonzero(self.is_test):
            d = self.diag[i]
            if d not in seen:
                seen.add(d)
                out.append(d)
            if len(out) >= limit:
                break
        return out

    # ---------- search ----------

    def text_to_image(self, query, top_k=10, test_only=False, dedupe_exams=True):
        q = self.encode_query(query)
        sims = (self.image_embeds @ q).numpy()

        mask = self._scope_mask(test_only)
        sims = np.where(mask, sims, -np.inf)

        # ground truth only exists when the query is a diagnosis in the set
        correct = set(np.flatnonzero((self.diag == query) & mask).tolist())

        order = np.argsort(-sims)
        results, seen_exams = [], set()
        for row in order:
            if not np.isfinite(sims[row]):
                break
            if dedupe_exams:
                exam = self.meta.iloc[row].ExamID
                if exam in seen_exams:
                    continue
                seen_exams.add(exam)
            results.append((int(row), float(sims[row]), int(row) in correct))
            if len(results) >= top_k:
                break
        return results, len(correct)

    def image_to_text(self, row, top_k=10, test_only=False):
        q = self.image_embeds[row]
        pool = self.text_pool_idx
        if test_only:
            pool = np.array([i for i in pool if self.is_test[i]])

        sims = (self.text_embeds[pool] @ q).numpy()
        truth = self.diag[row]

        order = np.argsort(-sims)[:top_k]
        return [(self.diag[pool[j]], float(sims[j]), bool(self.diag[pool[j]] == truth))
                for j in order], truth

    def row_info(self, row):
        m = self.meta.iloc[row]
        return {'id': int(m.id), 'modality': m.Modality, 'exam': int(m.ExamID),
                'split': m.split, 'n_views': len(self.views(row)),
                'diagnosis': m.DIAGNOSIS_TRUNCATED}

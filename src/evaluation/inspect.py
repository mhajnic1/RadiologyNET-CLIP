import os
import random
import textwrap

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from src.data.paths import get_image_dir, list_slices


def pick_examples(image_embeds, text_embeds, metadata, n_hits=2, n_misses=2, k=5, seed=42):
    diag = metadata['DIAGNOSIS_TRUNCATED'].values
    seen = {}
    query_indices = []
    for i, d in enumerate(diag):
        if d not in seen:
            seen[d] = i
            query_indices.append(i)

    query_embeds = text_embeds[query_indices]
    similarity = query_embeds @ image_embeds.T
    topk = similarity.topk(k, dim=1).indices

    hits, misses = [], []
    for qi, orig_i in enumerate(query_indices):
        correct = set(np.where(diag == diag[orig_i])[0].tolist())
        ranked = topk[qi].tolist()
        is_hit = any(idx in correct for idx in ranked)
        (hits if is_hit else misses).append((orig_i, ranked, correct))

    random.seed(seed)
    sampled_hits = random.sample(hits, min(n_hits, len(hits)))
    sampled_misses = random.sample(misses, min(n_misses, len(misses)))
    return sampled_hits, sampled_misses


def render_labelled_examples(rows, metadata, images_root, out_path, k=5, wrap_width=130):
    # same mosaic as render_examples, but each row carries its own caption so two
    # models can be put side by side in one figure and still be told apart
    n = len(rows)

    captions = []
    for label, (query_idx, _, _) in rows:
        query_text = metadata.iloc[query_idx].DIAGNOSIS_TRUNCATED
        captions.append(f"{label}\n" + textwrap.fill(f"Q: {query_text}", width=wrap_width))
    max_lines = max(c.count('\n') + 1 for c in captions)
    row_height = 3.0 + 0.22 * max_lines

    fig, axes = plt.subplots(n, k, figsize=(k * 2.5, n * row_height))
    if n == 1:
        axes = axes[None, :]

    for row, (_, (query_idx, ranked, correct)) in enumerate(rows):
        for col in range(k):
            img_idx = ranked[col]
            m = metadata.iloc[img_idx]
            folder = get_image_dir(images_root, m.id, m.Modality)
            slices = list_slices(images_root, m.id, m.Modality)
            img = Image.open(os.path.join(folder, slices[0])).convert('RGB')

            ax = axes[row, col]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            border = 'limegreen' if img_idx in correct else 'red'
            for spine in ax.spines.values():
                spine.set_edgecolor(border)
                spine.set_linewidth(4)
            ax.set_xlabel(f"rank {col+1}\n{m.Modality}", fontsize=7)

        axes[row, 0].annotate(
            captions[row], xy=(0, 1.10), xycoords='axes fraction',
            fontsize=8, ha='left', va='bottom', family='monospace',
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def render_examples(examples, metadata, images_root, out_path, k=5, wrap_width=140):
    n = len(examples)

    wrapped_queries = []
    for query_idx, _, _ in examples:
        query_text = metadata.iloc[query_idx].DIAGNOSIS_TRUNCATED
        wrapped_queries.append(textwrap.fill(f"Q: {query_text}", width=wrap_width))
    max_lines = max(w.count('\n') + 1 for w in wrapped_queries)
    row_height = 2.9 + 0.22 * max_lines

    fig, axes = plt.subplots(n, k, figsize=(k * 2.5, n * row_height))
    if n == 1:
        axes = axes[None, :]

    for row, (query_idx, ranked, correct) in enumerate(examples):
        for col in range(k):
            img_idx = ranked[col]
            m = metadata.iloc[img_idx]
            folder = get_image_dir(images_root, m.id, m.Modality)
            slices = list_slices(images_root, m.id, m.Modality)
            img = Image.open(os.path.join(folder, slices[0])).convert('RGB')

            ax = axes[row, col]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            border = 'limegreen' if img_idx in correct else 'red'
            for spine in ax.spines.values():
                spine.set_edgecolor(border)
                spine.set_linewidth(4)
            ax.set_xlabel(f"rank {col+1}\n{m.Modality}", fontsize=7)

        axes[row, 0].annotate(
            wrapped_queries[row], xy=(0, 1.12), xycoords='axes fraction',
            fontsize=8, ha='left', va='bottom',
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)

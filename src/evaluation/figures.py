"""Thesis figures, generated from the saved results and cached embeddings.

Everything here is numbers only, no patient images, so the output goes to
results/plots/ which is tracked, unlike results/figures/ which holds the
retrieval mosaics and is gitignored.
"""

import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.evaluation.plotstyle import save_figure

OUT_DIR = 'results/plots'

# one colour per model, kept identical across every figure so the reader can track
# a model from one plot to the next
COLOURS = {
    'clip_zs': '#b0b0b0',
    'clip_ft': '#5b8db8',
    'bmc_zs': '#e0a458',
    'bmc_ft': '#2e6f4e',
}
LABELS = {
    'clip_zs': 'CLIP, zero-shot',
    'clip_ft': 'CLIP, fine-tuned',
    'bmc_zs': 'BiomedCLIP, zero-shot',
    'bmc_ft': 'BiomedCLIP, fine-tuned',
}
RESULT_FILES = {
    'clip_zs': 'results/phase11_clip_zeroshot.json',
    'clip_ft': 'results/phase11_clip_finetuned.json',
    'bmc_zs': 'results/phase9_zeroshot_det.json',
    'bmc_ft': 'results/phase11_biomedclip_v1.json',
}


def _save(fig, name):
    return save_figure(fig, name, OUT_DIR)


def _load_results():
    return {k: json.load(open(p)) for k, p in RESULT_FILES.items()}


# --------------------------------------------------------------------------- 1

def fig_training_dynamics(history='results/tuning/v1_baseline_history.json'):
    """Validation loss and validation Recall@10 pulling apart over epochs."""
    h = json.load(open(history))
    ep = [r['epoch'] for r in h]
    train = [r['train_loss'] for r in h]
    val = [r['val_loss'] for r in h]
    rec = [r['val_recall_score'] for r in h]

    loss_best = int(np.argmin(val))
    rec_best = int(np.argmax(rec))

    fig, (a, b) = plt.subplots(2, 1, figsize=(7.8, 6.6), sharex=True,
                               gridspec_kw={'hspace': 0.14})

    # separate axes: training loss spans ~1.0-2.0 while validation moves only
    # 2.12-2.14, so on a shared scale the validation rise is invisible
    a.plot(ep, train, 'o-', color='#888', label='training loss (left)')
    a.set_ylabel('training loss', color='#888')
    a.tick_params(axis='y', labelcolor='#888')

    a2 = a.twinx()
    a2.plot(ep, val, 'o-', color='#c0392b', label='validation loss (right)')
    a2.set_ylabel('validation loss', color='#c0392b')
    a2.tick_params(axis='y', labelcolor='#c0392b')
    a2.axvline(loss_best, color='#c0392b', ls=':', lw=1.4)
    a2.annotate(f'lowest validation loss, epoch {loss_best}',
                xy=(loss_best, val[loss_best]),
                xytext=(loss_best + 0.45, val[loss_best] + 0.006),
                fontsize=8.5, color='#c0392b',
                arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1))

    lines = a.get_lines() + a2.get_lines()[:1]
    a.legend(lines, [l.get_label() for l in lines], frameon=False, fontsize=9, loc='center right')
    a.set_title('Validation loss stops tracking retrieval quality after epoch 3', fontsize=11.5)

    b.plot(ep, rec, 'o-', color='#2e6f4e', label='validation Recall@10')
    b.axvline(loss_best, color='#c0392b', ls=':', lw=1.4)
    b.axvline(rec_best, color='#2e6f4e', ls=':', lw=1.4)
    b.annotate(f'best retrieval, epoch {rec_best}',
               xy=(rec_best, rec[rec_best]), xytext=(rec_best - 2.9, rec[rec_best] + 0.002),
               fontsize=8.5, color='#2e6f4e',
               arrowprops=dict(arrowstyle='->', color='#2e6f4e', lw=1))
    gain = rec[rec_best] - rec[loss_best]
    b.annotate('', xy=(loss_best, rec[loss_best]), xytext=(loss_best, rec[rec_best]),
               arrowprops=dict(arrowstyle='<->', color='#444', lw=1.2))
    b.text(loss_best + 0.15, (rec[loss_best] + rec[rec_best]) / 2,
           f'+{gain:.4f} Recall@10\nlost by selecting on loss',
           fontsize=8.8, color='#444', va='center')
    b.set_ylabel('Recall@10 (mean of both directions)')
    b.set_xlabel('epoch')
    b.legend(frameon=False, fontsize=9, loc='lower right')

    for ax in (a, b):
        ax.grid(alpha=0.25, ls='--')
        ax.set_xticks(ep)
    _save(fig, '01_training_dynamics.png')


# --------------------------------------------------------------------------- 2

def fig_recall_at_k():
    """Recall@1/5/10 for all four models, both retrieval directions."""
    r = _load_results()
    ks = ['1', '5', '10']
    x = np.arange(len(ks))
    width = 0.2

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, key, title, pool in [
            (axes[0], 'text_to_image_recall', 'Diagnosis to image', 3046),
            (axes[1], 'image_to_text_recall', 'Image to diagnosis', 999)]:
        for i, m in enumerate(['clip_zs', 'clip_ft', 'bmc_zs', 'bmc_ft']):
            vals = [r[m][key][k] * 100 for k in ks]
            bars = ax.bar(x + (i - 1.5) * width, vals, width,
                          color=COLOURS[m], label=LABELS[m])
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.8, f'{v:.1f}',
                        ha='center', fontsize=7.2)
        chance = [100 * int(k) / pool for k in ks]
        ax.plot(x, chance, 'k_', ms=26, mew=1.6, label='random chance')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Recall@{k}' for k in ks])
        ax.set_ylabel('% of queries with a correct hit')
        ax.set_title(f'{title}  (pool of {pool:,})', fontsize=11)
        ax.grid(axis='y', alpha=0.25, ls='--')
        ax.set_ylim(0, 68)
    axes[0].legend(frameon=False, fontsize=8.6, ncol=1, loc='upper left')
    fig.suptitle('Retrieval accuracy on the held-out test split (3,046 images)', fontsize=12)
    fig.tight_layout()
    _save(fig, '02_recall_at_k.png')


# --------------------------------------------------------------------------- 3

def fig_base_model_2x2():
    """Pretraining domain against fine-tuning, and how they interact."""
    r = _load_results()
    zs = [r['clip_zs']['text_to_image_recall']['10'] * 100,
          r['bmc_zs']['text_to_image_recall']['10'] * 100]
    ft = [r['clip_ft']['text_to_image_recall']['10'] * 100,
          r['bmc_ft']['text_to_image_recall']['10'] * 100]

    x = np.arange(2)
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    b1 = ax.bar(x - width / 2, zs, width, color='#c9c9c9', label='zero-shot')
    b2 = ax.bar(x + width / 2, ft, width, color='#2e6f4e', label='fine-tuned on RadiologyNET')

    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.7,
                    f'{bar.get_height():.1f}%', ha='center', fontsize=9)

    for i in range(2):
        ax.annotate('', xy=(i + width / 2, ft[i]), xytext=(i - width / 2, zs[i]),
                    arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.5))
        ax.text(i, max(ft[i], zs[i]) + 4.2, f'+{ft[i] - zs[i]:.1f} pts',
                ha='center', fontsize=9.5, color='#c0392b')

    ax.plot([0 - width / 2, 1 - width / 2], zs, ':', color='#666', lw=1.3)
    ax.plot([0 + width / 2, 1 + width / 2], ft, ':', color='#666', lw=1.3)
    ax.text(0.5, (zs[0] + zs[1]) / 2 - 4.5, f'+{zs[1] - zs[0]:.1f} pts\nfrom pretraining',
            ha='center', fontsize=8.8, color='#666')
    ax.text(0.5, (ft[0] + ft[1]) / 2 + 2.0, f'+{ft[1] - ft[0]:.1f} pts\nfrom pretraining',
            ha='center', fontsize=8.8, color='#666')

    ax.set_xticks(x)
    ax.set_xticklabels(['original CLIP\n(general-purpose)', 'BiomedCLIP\n(biomedical pretraining)'])
    ax.set_ylabel('Recall@10, diagnosis to image (%)')
    ax.set_title('Pretraining and fine-tuning are substitutes, not additive', fontsize=11.5)
    ax.legend(frameon=False, fontsize=9.5)
    ax.grid(axis='y', alpha=0.25, ls='--')
    ax.set_ylim(0, 56)
    _save(fig, '03_base_model_2x2.png')


# --------------------------------------------------------------------------- 4

def fig_tuning_variants(summary='results/tuning/summary.json',
                        baseline='results/tuning/baseline_best_pt_val.json'):
    """Every tuning attempt against the pre-declared noise bar."""
    s = json.load(open(summary))
    base = json.load(open(baseline))['recall_score']

    names, scores, kinds = [], [], []
    for k, v in s.items():
        if v.get('status') == 'ok':
            names.append(k.replace('_', ' '))
            scores.append(v['best_val_recall_score'])
            kinds.append('train')

    # the three post-hoc attempts need no retraining, they sit on top of a trained model
    sc = json.load(open('results/tuning/soup_csls_val.json'))
    names.append('model soup')
    scores.append(sc['soup_v1v2v4']['no_csls']['score'])
    kinds.append('post')

    csls = max((v['score'] for k, v in sc['v2_head_lr (current best)'].items()
                if k.startswith('csls_')))
    names.append('CSLS (best k)')
    scores.append(csls)
    kinds.append('post')

    mv = json.load(open('results/tuning/multiview_val.json'))
    names.append('multi-view pooling')
    scores.append(mv['max_views_8']['score'])
    kinds.append('post')

    se = 0.0091
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.axvspan(base - 2 * se, base + 2 * se, color='#c0392b', alpha=0.10,
               label='within 2 standard errors of the baseline (a tie)')
    ax.axvline(base, color='#c0392b', lw=1.6, label=f'phase 6/7 checkpoint ({base:.4f})')

    cols = ['#5b8db8' if k == 'train' else '#8e7cc3' for k in kinds]
    ax.barh(y, scores, height=0.62, color=cols)
    ax.errorbar(scores, y, xerr=2 * se, fmt='none', ecolor='#333', capsize=3, lw=1)

    # labels parked at a fixed column so they never collide with the error bars
    for i, v in enumerate(scores):
        ax.text(0.5475, i, f'{v:.4f}', va='center', ha='right', fontsize=8.8,
                family='monospace')

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('validation Recall@10 (mean of both directions)')
    ax.set_xlim(0.47, 0.549)
    ax.invert_yaxis()
    ax.set_title('Nine attempts to improve the model, measured against the noise bar',
                 fontsize=11.5)
    # the two post-hoc methods that poke past the band do not survive a closer look,
    # and the figure has to say so rather than let the bars imply otherwise
    ax.text(0.5, -0.30,
            'No training variant clears the bar. CSLS is the best of a five-value sweep, '
            'and multi-view pooling\nfails a paired McNemar test (p = 0.13). The model soup '
            'led on validation and then lost on test.',
            transform=ax.transAxes, ha='center', va='top', fontsize=8.2, color='#444')

    from matplotlib.patches import Patch
    handles, labels = ax.get_legend_handles_labels()
    handles += [Patch(color='#5b8db8'), Patch(color='#8e7cc3')]
    labels += ['training variant', 'post-hoc, no retraining']
    ax.legend(handles, labels, frameon=False, fontsize=8.6,
              loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2)
    ax.grid(axis='x', alpha=0.25, ls='--')
    _save(fig, '04_tuning_variants.png')


# --------------------------------------------------------------------------- 5

def fig_clustering_stability(path='results/phase11_clustering_stable.json'):
    """Why a single-seed ARI is not a measurement."""
    d = json.load(open(path))
    names = list(d)
    img = [d[n]['image_embeds'] for n in names]

    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    for i, s in enumerate(img):
        ax.plot([s['ari_min'], s['ari_max']], [i, i], color='#999', lw=6, alpha=0.55,
                solid_capstyle='butt')
        ax.errorbar(s['ari_mean'], i, xerr=s['ari_std'], fmt='o', color='#2e6f4e',
                    ms=6, capsize=4, lw=1.4)
        ax.text(s['ari_max'] + 0.012, i, f"{s['ari_mean']:.3f} +- {s['ari_std']:.3f}",
                va='center', fontsize=8.4)

    ax.set_yticks(y)
    ax.set_yticklabels([n.replace('     ', ' ').replace('  ', ' ') for n in names], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Adjusted Rand Index against modality, image embeddings')
    ax.set_xlim(0.2, 1.0)
    ax.set_title('Grey bar = full range over 20 k-means seeds, dot = mean', fontsize=11)
    ax.grid(axis='x', alpha=0.25, ls='--')
    _save(fig, '05_clustering_stability.png')


# --------------------------------------------------------------------------- 6

def fig_embedding_tsne(seed=0, sample=1800):
    """Image embedding space before and after fine-tuning, coloured by modality."""
    pairs = [('bmc_zs', 'data/embeddings/zeroshot_det_test.pt'),
             ('bmc_ft', 'data/embeddings/v1_baseline_test.pt')]

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.2))
    rng = np.random.default_rng(seed)

    for ax, (key, path) in zip(axes, pairs):
        c = torch.load(path, weights_only=False)
        E = c['image_embeds'].numpy()
        mod = np.asarray(c['metadata']['Modality'].values, dtype=object)

        idx = rng.choice(len(E), size=min(sample, len(E)), replace=False)
        E, mod = E[idx], mod[idx]

        Z = PCA(n_components=50, random_state=seed).fit_transform(E)
        Z = TSNE(n_components=2, perplexity=30, init='pca',
                 random_state=seed).fit_transform(Z)

        for m, col in zip(['CR', 'CT', 'MR', 'RF', 'XA'],
                          ['#4c72b0', '#dd8452', '#55a868', '#c44e52', '#8172b3']):
            sel = mod == m
            ax.scatter(Z[sel, 0], Z[sel, 1], s=5, c=col, label=m, alpha=0.65, linewidths=0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(LABELS[key], fontsize=11)

    axes[0].legend(frameon=False, fontsize=9, markerscale=2.6, title='modality')
    fig.suptitle('t-SNE of image embeddings, test split, coloured by imaging modality', fontsize=12)
    fig.tight_layout()
    _save(fig, '06_embedding_tsne.png')


# --------------------------------------------------------------------------- 7

def fig_similarity_distribution():
    """How well matched pairs separate from unmatched ones."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), sharey=True)
    rng = np.random.default_rng(0)

    for ax, (key, path) in zip(axes, [('bmc_zs', 'data/embeddings/zeroshot_det_test.pt'),
                                       ('bmc_ft', 'data/embeddings/v1_baseline_test.pt')]):
        c = torch.load(path, weights_only=False)
        img, txt = c['image_embeds'], c['text_embeds']
        matched = (img * txt).sum(dim=1).numpy()

        perm = rng.permutation(len(img))
        # avoid accidentally pairing a row with itself, or with a row sharing its text
        diag = np.asarray(c['metadata']['DIAGNOSIS_TRUNCATED'].values, dtype=object)
        keep = diag[perm] != diag
        mismatched = (img[perm][keep] * txt[keep]).sum(dim=1).numpy()

        bins = np.linspace(-0.1, 0.75, 70)
        ax.hist(mismatched, bins=bins, color='#c9c9c9', label='unmatched pairs', density=True)
        ax.hist(matched, bins=bins, color='#2e6f4e', alpha=0.75, label='true pairs', density=True)
        ax.axvline(matched.mean(), color='#2e6f4e', ls='--', lw=1.3)
        ax.axvline(mismatched.mean(), color='#666', ls='--', lw=1.3)
        sep = matched.mean() - mismatched.mean()
        ax.set_title(f'{LABELS[key]}\nmean separation {sep:.3f}', fontsize=10.5)
        ax.set_xlabel('cosine similarity')
        ax.grid(alpha=0.22, ls='--')

    axes[0].set_ylabel('density')
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle('Similarity of true image-diagnosis pairs against unmatched pairs', fontsize=12)
    fig.tight_layout()
    _save(fig, '07_similarity_distribution.png')


if __name__ == '__main__':
    print('generating thesis figures')
    fig_training_dynamics()
    fig_recall_at_k()
    fig_base_model_2x2()
    fig_tuning_variants()
    fig_clustering_stability()
    fig_similarity_distribution()
    fig_embedding_tsne()
    print('done')

"""Figures that show real scans, so these go to results/figures/, which is gitignored.

Kept separate from figures.py, which is numbers only and safe to track.
"""

import os
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.data.paths import get_image_dir, list_slices
from src.evaluation.plotstyle import save_figure

OUT_DIR = 'results/figures'
FT = 'data/embeddings/v2_head_lr_test.pt'
ZS = 'data/embeddings/zeroshot_det_test.pt'


def _save(fig, name):
    return save_figure(fig, name, OUT_DIR)


def _load(path):
    c = torch.load(path, weights_only=False)
    return c['image_embeds'], c['text_embeds'], c['metadata']


def _query_index(diag):
    seen, out = set(), []
    for i, d in enumerate(diag):
        if d not in seen:
            seen.add(d)
            out.append(i)
    return out


def _rank(img, txt, qrow, k=5):
    sims = (txt[qrow] @ img.T).numpy()
    return np.argsort(-sims)[:k], sims


def _first_view(meta, row, images_root='data/images'):
    m = meta.iloc[row]
    folder = get_image_dir(images_root, m.id, m.Modality)
    return os.path.join(folder, list_slices(images_root, m.id, m.Modality)[0])


def _draw_row(axes, meta, ranked, correct, images_root='data/images'):
    for ax, row in zip(axes, ranked):
        m = meta.iloc[row]
        ax.imshow(Image.open(_first_view(meta, row, images_root)).convert('RGB'))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor('limegreen' if row in correct else 'red')
            sp.set_linewidth(3.5)
        ax.set_xlabel(m.Modality, fontsize=8)


# --------------------------------------------------------------------------- A

def fig_same_query_both_models(k=5, out='case_same_query.png'):
    """One diagnosis, both models, so the difference is visible rather than tabular.

    The query is chosen programmatically: the fine-tuned model must hit inside the
    top k and zero-shot must miss, which is the case that shows what fine-tuning
    bought. Picking one by eye would be cherry-picking.
    """
    img_f, txt_f, meta = _load(FT)
    img_z, txt_z, _ = _load(ZS)
    diag = meta['DIAGNOSIS_TRUNCATED'].values

    candidates = []
    for qrow in _query_index(diag):
        correct = set(np.flatnonzero(diag == diag[qrow]).tolist())
        rf, _ = _rank(img_f, txt_f, qrow, k)
        rz, _ = _rank(img_z, txt_z, qrow, k)
        hit_f = any(r in correct for r in rf)
        hit_z = any(r in correct for r in rz)
        if hit_f and not hit_z:
            # prefer a query short enough to print without swallowing the figure
            candidates.append((len(diag[qrow]), qrow, correct, rf, rz))

    candidates.sort()
    _, qrow, correct, rf, rz = candidates[len(candidates) // 6]
    print(f'  {len(candidates)} queries hit for fine-tuned and miss for zero-shot; '
          f'using row {qrow}')

    fig, axes = plt.subplots(2, k, figsize=(k * 2.3, 6.4))
    _draw_row(axes[0], meta, rz, correct)
    _draw_row(axes[1], meta, rf, correct)

    for ax_row, label, col in [(axes[0], 'BiomedCLIP, zero-shot', '#b06000'),
                               (axes[1], 'BiomedCLIP, fine-tuned', '#2e6f4e')]:
        ax_row[0].annotate(label, xy=(-0.06, 0.5), xycoords='axes fraction',
                           rotation=90, ha='right', va='center', fontsize=10, color=col)

    query = textwrap.fill('Query: ' + str(meta.iloc[qrow].DIAGNOSIS_TRUNCATED), width=118)
    fig.suptitle(query + '\n\ngreen = correct by exact diagnosis match, red = incorrect',
                 fontsize=9, y=1.02, ha='center')
    fig.tight_layout()
    _save(fig, out)


# --------------------------------------------------------------------------- B

def fig_modality_grid(out='case_modality_grid.png', seed=3):
    """One representative image per modality, for the data chapter."""
    _, _, meta = _load(FT)
    rng = np.random.default_rng(seed)
    mods = ['CR', 'CT', 'MR', 'RF', 'XA']
    names = {'CR': 'Computed Radiography', 'CT': 'Computed Tomography',
             'MR': 'Magnetic Resonance', 'RF': 'Radio Fluoroscopy',
             'XA': 'X-ray Angiography'}

    fig, axes = plt.subplots(1, 5, figsize=(13, 3.1))
    for ax, m in zip(axes, mods):
        rows = np.flatnonzero(np.asarray(meta['Modality'].values, dtype=object) == m)
        row = int(rng.choice(rows))
        ax.imshow(Image.open(_first_view(meta, row)).convert('RGB'))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(m, fontsize=12)
        ax.set_xlabel(names[m], fontsize=8.5)

    fig.suptitle('The five imaging modalities in RadiologyNET, one example each',
                 fontsize=12)
    fig.tight_layout()
    _save(fig, out)


# --------------------------------------------------------------------------- C

def fig_multifile_structure(out='case_multifile_structure.png'):
    """What the extra PNGs per image actually are: CT windowings, XA frames."""
    _, _, meta = _load(FT)
    mod = np.asarray(meta['Modality'].values, dtype=object)

    def views(row):
        m = meta.iloc[row]
        folder = get_image_dir('data/images', m.id, m.Modality)
        return [os.path.join(folder, n)
                for n in list_slices('data/images', m.id, m.Modality)]

    ct_row = next(r for r in np.flatnonzero(mod == 'CT') if len(views(r)) == 2)

    # pick the XA study whose frames actually change. many sequences are near static,
    # and one of those would illustrate nothing about the files being temporal
    best, best_var = None, -1.0
    for r in [r for r in np.flatnonzero(mod == 'XA') if len(views(r)) >= 20][:60]:
        v = views(r)
        idx = np.linspace(0, len(v) - 1, 5).round().astype(int)
        means = [np.asarray(Image.open(v[j]).convert('L')).mean() for j in idx]
        if np.std(means) > best_var:
            best, best_var = int(r), float(np.std(means))
    xa_row = best
    print(f'  XA study chosen for frame variation (sd of frame means {best_var:.1f})')

    ct = views(ct_row)
    xa = views(xa_row)
    picks = np.linspace(0, len(xa) - 1, 5).round().astype(int)

    fig = plt.figure(figsize=(12.4, 6.6))
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1])

    ax_ct = top.subplots(1, 2)
    top.suptitle('CT: the two files are the same anatomical slice under two windowings',
                 fontsize=11)
    for ax, p in zip(ax_ct, ct):
        arr = np.asarray(Image.open(p).convert('L'))
        ax.imshow(arr, cmap='gray', vmin=0, vmax=255)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(f'{os.path.basename(p)}   mean intensity {arr.mean():.0f}', fontsize=9)

    ax_xa = bottom.subplots(1, 5)
    bottom.suptitle(f'XA: the extra files are temporal frames of a fluoroscopy sequence '
                    f'({len(xa)} frames in this study)', fontsize=11)
    for ax, j in zip(ax_xa, picks):
        arr = np.asarray(Image.open(xa[j]).convert('L'))
        ax.imshow(arr, cmap='gray', vmin=0, vmax=255)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(f'frame {j}   mean {arr.mean():.0f}', fontsize=8.5)

    _save(fig, out)


# --------------------------------------------------------------------------- D

def fig_boilerplate_miss(k=5, out='case_boilerplate_miss.png'):
    """A miss where every retrieved image is a reasonable answer.

    Picks the shortest missed query, since short reports are the generic
    normal-finding ones where many images would be equally defensible.
    """
    img, txt, meta = _load(FT)
    diag = meta['DIAGNOSIS_TRUNCATED'].values

    misses = []
    for qrow in _query_index(diag):
        correct = set(np.flatnonzero(diag == diag[qrow]).tolist())
        ranked, sims = _rank(img, txt, qrow, k)
        if not any(r in correct for r in ranked):
            misses.append((len(diag[qrow]), qrow, correct, ranked, sims))

    misses.sort()
    _, qrow, correct, ranked, sims = misses[0]
    same_mod = sum(1 for r in ranked
                   if meta.iloc[r].Modality == meta.iloc[qrow].Modality)
    print(f'  shortest missed query is row {qrow}, '
          f'{same_mod}/{k} retrieved share its modality')

    fig, axes = plt.subplots(1, k, figsize=(k * 2.3, 3.4))
    for ax, r in zip(axes, ranked):
        m = meta.iloc[r]
        ax.imshow(Image.open(_first_view(meta, r)).convert('RGB'))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor('red')
            sp.set_linewidth(3.5)
        ax.set_xlabel(f'{m.Modality}   {sims[r]:.3f}', fontsize=8)

    # wrapped narrower than the panel is wide so the bigger type still fits on one
    # figure width. at 12 pt about 95 characters is what the 11.5 inches hold
    query = textwrap.fill('Query: ' + str(meta.iloc[qrow].DIAGNOSIS_TRUNCATED), width=95)
    fig.suptitle(query + '\n\nAll five are plausible answers, none is the one exact '
                         'text match, so all count as wrong',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, out)


# --------------------------------------------------------------------------- E

def fig_manual_inspection(seed=42, out=None, split=False):
    """The qualitative check from phase 7: two hits and two misses, top 5 each.

    Regenerated against v2_head_lr with deterministic first-slice selection. The
    original PNGs came from best.pt while the random-slice bug was still live, so
    the rows there are not reproducible.
    """
    from src.evaluation.inspect import pick_examples, render_examples

    out = out or f'manual_inspection{"" if seed == 42 else f"_seed{seed}"}.png'
    img, txt, meta = _load(FT)
    hits, misses = pick_examples(img, txt, meta, n_hits=2, n_misses=2, k=5, seed=seed)
    print(f'  seed {seed}: {len(hits)} hits + {len(misses)} misses')
    render_examples(hits + misses, meta, 'data/images', os.path.join(OUT_DIR, out),
                    k=5, split=split)


# --------------------------------------------------------------------------- F

def fig_finetuned_vs_zeroshot(seed=42, n_queries=2, k=5, out=None):
    """Same queries through both models, one pair of rows per query.

    Queries are sampled from the ones where the two models disagree, so the figure
    shows something. Which of those it picks is down to the seed.
    """
    from src.evaluation.inspect import render_labelled_examples

    out = out or f'finetuned_vs_zeroshot{"" if seed == 42 else f"_seed{seed}"}.png'
    img_f, txt_f, meta = _load(FT)
    img_z, txt_z, _ = _load(ZS)
    diag = meta['DIAGNOSIS_TRUNCATED'].values

    disagree = []
    for qrow in _query_index(diag):
        correct = set(np.flatnonzero(diag == diag[qrow]).tolist())
        rf, _ = _rank(img_f, txt_f, qrow, k)
        rz, _ = _rank(img_z, txt_z, qrow, k)
        if any(r in correct for r in rf) != any(r in correct for r in rz):
            disagree.append((qrow, correct, list(rf), list(rz)))

    rng = np.random.default_rng(seed)
    picks = rng.choice(len(disagree), size=min(n_queries, len(disagree)), replace=False)
    print(f'  seed {seed}: {len(disagree)} queries where the models disagree, '
          f'showing {len(picks)}')

    rows = []
    for i in picks:
        qrow, correct, rf, rz = disagree[i]
        rows.append(('zero-shot BiomedCLIP', (qrow, rz, correct)))
        rows.append(('fine-tuned BiomedCLIP', (qrow, rf, correct)))

    render_labelled_examples(rows, meta, 'data/images', os.path.join(OUT_DIR, out), k=k)


# --------------------------------------------------------------------------- G

def fig_slice_audit(out='slice_audit.png', n_ct=3, n_xa=3, max_cols=6):
    """Audit of what the extra files per study contain.

    The question this answers is whether taking only the first file throws away
    real information. For CT the second file is the same slice re-windowed, for XA
    the extra files are frames of one sequence. Mean intensity is printed so the
    windowing difference is readable rather than something I assert.
    """
    _, _, meta = _load(FT)
    mod = np.asarray(meta['Modality'].values, dtype=object)

    def views(row):
        m = meta.iloc[row]
        folder = get_image_dir('data/images', m.id, m.Modality)
        return [os.path.join(folder, n)
                for n in list_slices('data/images', m.id, m.Modality)]

    ct_rows = [r for r in np.flatnonzero(mod == 'CT') if len(views(r)) > 1][:n_ct]
    xa_rows = [r for r in np.flatnonzero(mod == 'XA') if len(views(r)) >= 20][:n_xa]
    picked = [('CT', int(r)) for r in ct_rows] + [('XA', int(r)) for r in xa_rows]

    fig, axes = plt.subplots(len(picked), max_cols,
                             figsize=(max_cols * 2.0, len(picked) * 2.35))
    for ax in axes.ravel():
        ax.axis('off')

    for row, (label, r) in enumerate(picked):
        v = views(r)
        idx = (np.linspace(0, len(v) - 1, min(max_cols, len(v)))
               .round().astype(int))
        for col, j in enumerate(idx):
            arr = np.asarray(Image.open(v[j]).convert('L'))
            ax = axes[row, col]
            ax.axis('on')
            ax.imshow(arr, cmap='gray', vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor('#4477aa' if col == 0 else '#bbbbbb')
                sp.set_linewidth(3 if col == 0 else 1)
            ax.set_xlabel(f'file {j}   mean {arr.mean():.0f}', fontsize=7.5)

        axes[row, 0].annotate(f'{label}  id {meta.iloc[r].id}\n{len(v)} files',
                              xy=(-0.10, 0.5), xycoords='axes fraction',
                              rotation=90, ha='right', va='center', fontsize=8)

    fig.suptitle('What the extra files per study contain. Blue border marks the '
                 'first file, the only one the evaluation uses.\n'
                 'CT: the same slice under different windowings. '
                 'XA: temporal frames of one fluoroscopy sequence.',
                 fontsize=10, y=1.0)
    fig.tight_layout()
    _save(fig, out)


if __name__ == '__main__':
    print('generating case figures (these contain real scans)')
    fig_same_query_both_models()
    fig_modality_grid()
    fig_multifile_structure()
    fig_boilerplate_miss()
    fig_slice_audit()
    for s in (42, 1, 2, 3, 4):
        fig_manual_inspection(seed=s)
        fig_finetuned_vs_zeroshot(seed=s)
    print('done')

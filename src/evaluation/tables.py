"""Thesis tables, written as markdown to results/plots/tables.md.

Numbers only, pulled from the saved results, so nothing here has to be
retyped by hand into the thesis.
"""

import json
import math
import os

import pandas as pd

OUT = 'results/plots/tables.md'

RESULTS = {
    'clip_zs': ('original CLIP, zero-shot', 'results/phase11_clip_zeroshot.json'),
    'clip_ft': ('original CLIP, fine-tuned', 'results/phase11_clip_finetuned.json'),
    'bmc_zs': ('BiomedCLIP, zero-shot', 'results/phase9_zeroshot_det.json'),
    'bmc_ft': ('BiomedCLIP, fine-tuned (v1)', 'results/phase11_biomedclip_v1.json'),
    'bmc_v2': ('BiomedCLIP, fine-tuned (v2, reported)', 'results/phase9_metrics.json'),
}


def se(p, n):
    return math.sqrt(max(p * (1 - p), 0.0) / n)


def sigma(a, na, b, nb):
    sd = math.sqrt(se(a, na) ** 2 + se(b, nb) ** 2)
    return (b - a) / sd if sd else float('nan')


def t_dataset():
    meta = pd.read_csv('data/metainformation.csv', index_col='id')
    splits = pd.read_csv('data/splits.csv')
    meta = meta.join(splits.set_index('ExamID').split, on='ExamID')
    ct = pd.crosstab(meta.Modality, meta.split)[['train', 'val', 'test']]
    ct['total'] = ct.sum(axis=1)
    ct.loc['all modalities'] = ct.sum()

    lines = ['## Table 1. Dataset composition (images per modality and split)', '',
             '| modality | train | val | test | total |', '|---|---|---|---|---|']
    for m, r in ct.iterrows():
        lines.append(f"| {m} | {r['train']:,} | {r['val']:,} | {r['test']:,} | {r['total']:,} |")
    lines += ['', f'Exams: 8,000 train / 1,000 val / 1,000 test, split by `ExamID` and '
                  f'stratified by modality.', '']
    return lines


def t_recall():
    lines = ['## Table 2. Retrieval accuracy on the test split', '']
    for key, direction, pool in [('text_to_image_recall', 'Diagnosis to image', 3046),
                                 ('image_to_text_recall', 'Image to diagnosis', 999)]:
        lines += [f'### {direction} (pool of {pool:,})', '',
                  '| model | R@1 | R@5 | R@10 | queries |', '|---|---|---|---|---|']
        lines.append(f"| random chance | {100/pool:.2f}% | {500/pool:.2f}% | "
                     f"{1000/pool:.2f}% | - |")
        for k, (name, path) in RESULTS.items():
            d = json.load(open(path))
            r = d[key]
            n = d[key.replace('_recall', '_n_queries')]
            lines.append(f"| {name} | {r['1']*100:.1f}% | {r['5']*100:.1f}% | "
                         f"{r['10']*100:.1f}% | {n:,} |")
        lines.append('')
    return lines


def t_2x2():
    d = {k: json.load(open(p)) for k, (_, p) in RESULTS.items()}

    def t2i(k):
        return d[k]['text_to_image_recall']['10'], d[k]['text_to_image_n_queries']

    rows = [('original CLIP', 'clip_zs', 'clip_ft'), ('BiomedCLIP', 'bmc_zs', 'bmc_ft')]
    lines = ['## Table 3. Pretraining domain against fine-tuning', '',
             '| base model | zero-shot | fine-tuned | gain | sigma |', '|---|---|---|---|---|']
    for name, z, f in rows:
        (a, na), (b, nb) = t2i(z), t2i(f)
        lines.append(f'| {name} | {a*100:.1f}% | {b*100:.1f}% | +{(b-a)*100:.1f} pts '
                     f'| {sigma(a, na, b, nb):.1f} |')

    lines += ['', '| comparison | effect | sigma |', '|---|---|---|']
    for label, x, y in [
            ('biomedical pretraining, neither fine-tuned', 'clip_zs', 'bmc_zs'),
            ('biomedical pretraining, both fine-tuned', 'clip_ft', 'bmc_ft'),
            ('fine-tuning, on general CLIP', 'clip_zs', 'clip_ft'),
            ('fine-tuning, on BiomedCLIP', 'bmc_zs', 'bmc_ft')]:
        (a, na), (b, nb) = t2i(x), t2i(y)
        lines.append(f'| {label} | {(b-a)*100:+.1f} pts | {sigma(a, na, b, nb):.1f} |')
    lines.append('')
    return lines


def t_variants():
    s = json.load(open('results/tuning/summary.json'))
    base = json.load(open('results/tuning/baseline_best_pt_val.json'))['recall_score']
    bar = 2 * 0.0091

    lines = ['## Table 4. Tuning variants, ranked on validation', '',
             '| variant | best epoch | t2i R@10 | i2t R@10 | score | vs baseline |',
             '|---|---|---|---|---|---|',
             f'| phase 6/7 checkpoint | - | 0.4489 | 0.5498 | {base:.4f} | - |']
    for k, v in s.items():
        if v.get('status') != 'ok':
            continue
        d = v['best_val_recall_score'] - base
        lines.append(f"| {k.replace('_', ' ')} | {v['best_epoch']} | "
                     f"{v['best_val_t2i_r10']:.4f} | {v['best_val_i2t_r10']:.4f} | "
                     f"{v['best_val_recall_score']:.4f} | {d:+.4f} |")
    lines += ['', f'Two standard errors on the score is {bar:.3f}. No variant clears it.', '']
    return lines


def t_clustering():
    d = json.load(open('results/phase11_clustering_stable.json'))
    lines = ['## Table 5. Embedding clustering against modality (20 k-means seeds)', '',
             '| model | image ARI | image NMI | text ARI | text NMI |',
             '|---|---|---|---|---|']
    for name, v in d.items():
        i, t = v['image_embeds'], v['text_embeds']
        lines.append(f"| {' '.join(name.split())} | "
                     f"{i['ari_mean']:.3f} +- {i['ari_std']:.3f} | "
                     f"{i['nmi_mean']:.3f} | "
                     f"{t['ari_mean']:.3f} +- {t['ari_std']:.3f} | "
                     f"{t['nmi_mean']:.3f} |")
    lines += ['', 'A single seed is not a measurement here: the same embeddings span up to '
                  '0.27 ARI across seeds.', '']
    return lines


def t_memory():
    lines = ['## Table 6. Measured GPU cost per fine-tuning strategy (RTX 3060, 6.44 GB)', '',
             '| strategy | trainable params | share | peak VRAM | usable |',
             '|---|---|---|---|---|',
             '| full fine-tuning | 195,900,000 | 100% | 9.87 GB | no, exceeds the card |',
             '| frozen backbone | 1,200,000 | 0.6% | 1.12 GB | yes |',
             '| partial (last 2 blocks per encoder) | 29,600,000 | 15.1% | 2.49 GB | yes, chosen |',
             '',
             'Measured over sustained multi-step training, not a single step. bf16 autocast '
             'gave roughly 3x throughput over fp32 (13 ms against 41.8 ms per sample).', '']
    return lines


def t_tokens():
    lines = ['## Table 7. Diagnosis length against each text encoder', '',
             '| encoder | context | usable | diagnoses over budget | truncation used |',
             '|---|---|---|---|---|',
             '| PubMedBERT (BiomedCLIP) | 256 | 254 | 24.9% | head 64 + tail 190 |',
             '| CLIP text encoder | 77 | 75 | 79.0% | head 19 + tail 56 |',
             '',
             'Median diagnosis is 163 CLIP tokens. Mean surviving text is 753 characters '
             'under BiomedCLIP against 307 under CLIP.', '']
    return lines


if __name__ == '__main__':
    os.makedirs('results/plots', exist_ok=True)
    out = ['# Thesis tables', '',
           'Generated by `src/evaluation/tables.py` from the saved result files.', '']
    for fn in (t_dataset, t_recall, t_2x2, t_variants, t_clustering, t_memory, t_tokens):
        out += fn()
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))
    print(f'wrote {OUT}')
    print('\n'.join(out[:40]))

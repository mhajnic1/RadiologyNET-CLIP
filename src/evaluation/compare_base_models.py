import json
import math
import os

# the 2x2: pretraining domain (general vs biomedical) against adaptation (none vs
# fine-tuned on RadiologyNET). row difference is what domain pretraining buys,
# column difference is what fine-tuning buys
# BiomedCLIP fine-tuned uses v1_baseline, not the v2_head_lr that phase 9 reported.
# v1 is config-identical to the CLIP run (head_lr_mult=1.0), v2 used 10.0, so v1 is the
# only apples-to-apples cell. phase 9 measured head_lr_mult as a null (+0.0009, a tenth
# of the noise bar) so it barely matters, but "barely matters" is an inference and the
# clean comparison costs nothing
CELLS = [
    ('original CLIP', 'zero-shot', 'results/phase11_clip_zeroshot.json'),
    ('original CLIP', 'fine-tuned', 'results/phase11_clip_finetuned.json'),
    ('BiomedCLIP', 'zero-shot', 'results/phase9_zeroshot_det.json'),
    ('BiomedCLIP', 'fine-tuned', 'results/phase11_biomedclip_v1.json'),
]

# kept only so the write-up can state how little the confound was worth
V2_REPORTED = 'results/phase9_metrics.json'


def se(p, n):
    return math.sqrt(max(p * (1 - p), 0.0) / n)


def load():
    out = {}
    for base, cond, path in CELLS:
        if not os.path.exists(path):
            out[(base, cond)] = None
            continue
        d = json.load(open(path))
        out[(base, cond)] = {
            't2i': d['text_to_image_recall']['10'],
            'i2t': d['image_to_text_recall']['10'],
            'n_t2i': d['text_to_image_n_queries'],
            'n_i2t': d['image_to_text_n_queries'],
            'score': d['recall_score'],
            'img_ari': d['clustering_image_embeds']['ari'],
            'txt_ari': d['clustering_text_embeds']['ari'],
        }
    return out


def gap(a, b, key, nkey):
    """difference between two cells in standard errors, for the metric named by key"""
    if not a or not b:
        return None
    sd = math.sqrt(se(a[key], a[nkey]) ** 2 + se(b[key], b[nkey]) ** 2)
    return (b[key] - a[key]) / sd if sd else None


def report():
    r = load()
    missing = [f'{b} {c}' for (b, c), v in r.items() if v is None]
    if missing:
        print('not computed yet:', ', '.join(missing), '\n')

    print('text to image Recall@10 (the task)')
    print(f"{'':<16}{'zero-shot':>12}{'fine-tuned':>13}{'gain':>9}{'sigma':>8}")
    for base in ('original CLIP', 'BiomedCLIP'):
        z, f = r[(base, 'zero-shot')], r[(base, 'fine-tuned')]
        if z and f:
            g = f['t2i'] - z['t2i']
            s = gap(z, f, 't2i', 'n_t2i')
            print(f"{base:<16}{z['t2i']:>11.1%}{f['t2i']:>13.1%}{g:>+9.1%}{s:>8.1f}")
        elif z:
            print(f"{base:<16}{z['t2i']:>11.1%}{'-':>13}")

    print('\nwhat each axis buys, on text to image Recall@10')
    zc, zb = r[('original CLIP', 'zero-shot')], r[('BiomedCLIP', 'zero-shot')]
    if zc and zb:
        print(f"  biomedical pretraining, no fine-tuning : {zb['t2i'] - zc['t2i']:+.1%} "
              f"({gap(zc, zb, 't2i', 'n_t2i'):.1f} sigma)")
    fc, fb = r[('original CLIP', 'fine-tuned')], r[('BiomedCLIP', 'fine-tuned')]
    if fc and fb:
        print(f"  biomedical pretraining, both fine-tuned: {fb['t2i'] - fc['t2i']:+.1%} "
              f"({gap(fc, fb, 't2i', 'n_t2i'):.1f} sigma)")
    if zc and fc:
        print(f"  fine-tuning on top of general CLIP     : {fc['t2i'] - zc['t2i']:+.1%} "
              f"({gap(zc, fc, 't2i', 'n_t2i'):.1f} sigma)")
    if zb and fb:
        print(f"  fine-tuning on top of BiomedCLIP       : {fb['t2i'] - zb['t2i']:+.1%} "
              f"({gap(zb, fb, 't2i', 'n_t2i'):.1f} sigma)")

    if os.path.exists(V2_REPORTED):
        v2 = json.load(open(V2_REPORTED))
        v1 = r[('BiomedCLIP', 'fine-tuned')]
        if v1:
            print(f"\nhow much the head_lr_mult confound was actually worth on test:")
            print(f"  v1_baseline (head_lr_mult 1.0, matches the CLIP run): {v1['t2i']:.4f}")
            print(f"  v2_head_lr  (head_lr_mult 10.0, phase 9's reported) : {v2['text_to_image_recall']['10']:.4f}")
            print(f"  difference: {v2['text_to_image_recall']['10'] - v1['t2i']:+.4f}")

    print('\nfull table')
    print(f"{'model':<16}{'condition':<12}{'t2i@10':>8}{'i2t@10':>8}{'score':>8}{'imgARI':>8}{'txtARI':>8}")
    for base, cond, _ in CELLS:
        v = r[(base, cond)]
        if v:
            print(f"{base:<16}{cond:<12}{v['t2i']:>8.4f}{v['i2t']:>8.4f}"
                  f"{v['score']:>8.4f}{v['img_ari']:>8.4f}{v['txt_ari']:>8.4f}")
    return r


if __name__ == '__main__':
    report()

import json
import math
import os


def recall_stderr(recall, n_queries):
    # Recall@K is a proportion over n independent queries, so the usual binomial
    # standard error applies. without this it's far too easy to read a 1 point
    # difference between two variants as a real improvement
    return math.sqrt(max(recall * (1 - recall), 0.0) / n_queries)


def combined_stderr(t2i, n_t2i, i2t, n_i2t):
    # the score is the mean of two independent proportions, so errors add in quadrature
    return 0.5 * math.sqrt(recall_stderr(t2i, n_t2i) ** 2 + recall_stderr(i2t, n_i2t) ** 2)


def compare(results_dir='results/tuning', n_t2i=998, n_i2t=2890,
            baseline_path='results/tuning/baseline_best_pt_val.json'):
    rows = []

    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            b = json.load(f)
        rows.append({
            'name': 'phase 6/7 checkpoint (baseline)',
            'epoch': None,
            't2i_r10': b['text_to_image_recall']['10'],
            'i2t_r10': b['image_to_text_recall']['10'],
            'score': b['recall_score'],
        })

    summary_path = os.path.join(results_dir, 'summary.json')
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        for name, r in summary.items():
            if r.get('status') != 'ok':
                rows.append({'name': name, 'epoch': None, 't2i_r10': None,
                             'i2t_r10': None, 'score': None})
                continue
            rows.append({
                'name': name,
                'epoch': r['best_epoch'],
                't2i_r10': r['best_val_t2i_r10'],
                'i2t_r10': r['best_val_i2t_r10'],
                'score': r['best_val_recall_score'],
            })

    baseline_score = rows[0]['score'] if rows else None

    print(f"{'variant':<34} {'ep':>3} {'t2i@10':>8} {'i2t@10':>8} {'score':>8} {'+-':>6} {'vs base':>9}")
    print('-' * 82)
    for r in rows:
        if r['score'] is None:
            print(f"{r['name']:<34} {'':>3} {'failed':>8}")
            continue
        se = combined_stderr(r['t2i_r10'], n_t2i, r['i2t_r10'], n_i2t)
        delta = r['score'] - baseline_score if baseline_score is not None else 0.0
        ep = '' if r['epoch'] is None else r['epoch']
        flag = ''
        if baseline_score is not None and r['name'] != rows[0]['name']:
            # 2 standard errors is the usual rough bar for "probably not just noise"
            flag = ' *' if abs(delta) > 2 * se else ''
        print(f"{r['name']:<34} {ep:>3} {r['t2i_r10']:>8.4f} {r['i2t_r10']:>8.4f} "
              f"{r['score']:>8.4f} {se:>6.4f} {delta:>+9.4f}{flag}")

    print('\n* = difference from baseline exceeds 2 standard errors')
    return rows


if __name__ == '__main__':
    compare()

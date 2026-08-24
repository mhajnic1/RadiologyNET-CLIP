import json
import os

from src.evaluation.clustering import (
    cluster_embeddings, evaluate_against_modality, stable_ari_nmi,
)
from src.evaluation.embed import (
    load_trained_model, load_zeroshot_model, load_zeroshot_original_clip,
    get_split_embeddings,
)
from src.evaluation.retrieval import (
    text_to_image_recall, image_to_text_recall, random_baseline_recall,
)


def full_eval(cache_name, checkpoint_path=None, split='test', out_path=None,
              n_clusters=5, num_workers=4, label=None, base_model='biomedclip'):
    # same pipeline phase 7 and 8 ran, just parameterised so any checkpoint goes through
    # exactly the same code path and the numbers stay comparable.
    # base_model also picks the text column, since the original CLIP only has 77 tokens
    # and needs its own truncation
    text_column = 'DIAGNOSIS_TRUNCATED_77' if base_model == 'clip' else 'DIAGNOSIS_TRUNCATED'

    if checkpoint_path:
        model, preprocess_val, tokenizer = load_trained_model(checkpoint_path, base_model)
    elif base_model == 'clip':
        model, preprocess_val, tokenizer = load_zeroshot_original_clip()
    else:
        model, preprocess_val, tokenizer = load_zeroshot_model()

    image_embeds, text_embeds, metadata = get_split_embeddings(
        split, model, preprocess_val, tokenizer, cache_name=cache_name,
        text_column=text_column,
    )

    t2i, t2i_n = text_to_image_recall(image_embeds, text_embeds, metadata)
    i2t, i2t_n = image_to_text_recall(image_embeds, text_embeds, metadata)

    # across 20 seeds, not one. a single seed swings by up to 0.27 ARI on some models,
    # which is larger than any difference between models, so single-seed numbers are
    # not a measurement. crosstabs still come from seed 42 for readability
    img_stable = stable_ari_nmi(image_embeds, metadata, n_clusters=n_clusters)
    txt_stable = stable_ari_nmi(text_embeds, metadata, n_clusters=n_clusters)

    img_clusters = cluster_embeddings(image_embeds, n_clusters=n_clusters)
    img_ari, img_nmi, img_crosstab = evaluate_against_modality(img_clusters, metadata)
    txt_clusters = cluster_embeddings(text_embeds, n_clusters=n_clusters)
    txt_ari, txt_nmi, txt_crosstab = evaluate_against_modality(txt_clusters, metadata)

    base_name = 'original CLIP (ViT-B-16-quickgelu, openai)' if base_model == 'clip' else 'BiomedCLIP'
    results = {
        'label': label or (checkpoint_path or f'zero-shot {base_name}'),
        'checkpoint': checkpoint_path or f'pretrained {base_name} (no fine-tuning)',
        'base_model': base_model,
        'text_column': text_column,
        'split': split,
        'n_images': int(image_embeds.shape[0]),
        'text_to_image_recall': t2i,
        'text_to_image_n_queries': t2i_n,
        'text_to_image_random_baseline': random_baseline_recall(image_embeds.shape[0]),
        'image_to_text_recall': i2t,
        'image_to_text_n_queries': i2t_n,
        'image_to_text_random_baseline': random_baseline_recall(t2i_n),
        'recall_score': (t2i[10] + i2t[10]) / 2,
        # the numbers to actually cite
        'clustering_image_stable': img_stable,
        'clustering_text_stable': txt_stable,
        # single seed 42, kept only so older result files stay comparable. unreliable,
        # see clustering_*_stable for the spread
        'clustering_image_embeds': {'ari': img_ari, 'nmi': img_nmi, 'n_clusters': n_clusters,
                                    'single_seed_unreliable': True},
        'clustering_text_embeds': {'ari': txt_ari, 'nmi': txt_nmi, 'n_clusters': n_clusters,
                                   'single_seed_unreliable': True},
    }

    print(json.dumps({k: v for k, v in results.items()}, indent=2), flush=True)
    print('\nimage embed clusters vs modality:\n', img_crosstab, flush=True)
    print('\ntext embed clusters vs modality:\n', txt_crosstab, flush=True)

    if out_path:
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)

    return results, img_crosstab, txt_crosstab


if __name__ == '__main__':
    import sys
    full_eval(cache_name=sys.argv[1],
              checkpoint_path=sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None,
              out_path=sys.argv[3] if len(sys.argv) > 3 else None,
              base_model=sys.argv[4] if len(sys.argv) > 4 else 'biomedclip')

import numpy as np


def _dedupe_by_text(diag):
    seen = {}
    indices = []
    for i, d in enumerate(diag):
        if d not in seen:
            seen[d] = i
            indices.append(i)
    return indices


def csls_adjust(similarity, k=10):
    # hubness correction. in cross modal retrieval a few gallery items sit close to
    # loads of unrelated queries and hoover up top-k slots, so subtract how popular
    # each gallery item is in general before ranking.
    # full CSLS also subtracts a query side term, but that's constant along a row and
    # so cannot change that query's ranking, only the gallery term matters here.
    # Conneau et al. 2018, arXiv:1710.04087
    k = min(k, similarity.shape[0])
    gallery_hubness = similarity.topk(k, dim=0).values.mean(dim=0)
    return 2 * similarity - gallery_hubness.unsqueeze(0)


def text_to_image_recall(image_embeds, text_embeds, metadata, ks=(1, 5, 10), csls_k=None):
    diag = metadata['DIAGNOSIS_TRUNCATED'].values
    query_indices = _dedupe_by_text(diag)

    query_embeds = text_embeds[query_indices]
    similarity = query_embeds @ image_embeds.T
    if csls_k:
        similarity = csls_adjust(similarity, csls_k)

    max_k = max(ks)
    topk_indices = similarity.topk(max_k, dim=1).indices

    hits = {k: 0 for k in ks}
    for qi, orig_i in enumerate(query_indices):
        correct = set(np.where(diag == diag[orig_i])[0].tolist())
        ranked = topk_indices[qi].tolist()
        for k in ks:
            if any(idx in correct for idx in ranked[:k]):
                hits[k] += 1

    n_queries = len(query_indices)
    return {k: hits[k] / n_queries for k in ks}, n_queries


def image_to_text_recall(image_embeds, text_embeds, metadata, ks=(1, 5, 10), csls_k=None):
    diag = metadata['DIAGNOSIS_TRUNCATED'].values
    pool_indices = _dedupe_by_text(diag)

    pool_embeds = text_embeds[pool_indices]
    pool_diag = diag[pool_indices]
    similarity = image_embeds @ pool_embeds.T
    if csls_k:
        similarity = csls_adjust(similarity, csls_k)

    max_k = max(ks)
    topk_indices = similarity.topk(max_k, dim=1).indices

    hits = {k: 0 for k in ks}
    for i in range(len(diag)):
        correct = set(np.where(pool_diag == diag[i])[0].tolist())
        ranked = topk_indices[i].tolist()
        for k in ks:
            if any(idx in correct for idx in ranked[:k]):
                hits[k] += 1

    n_queries = len(diag)
    return {k: hits[k] / n_queries for k in ks}, n_queries


def random_baseline_recall(pool_size, ks=(1, 5, 10)):
    return {k: min(1.0, k / pool_size) for k in ks}

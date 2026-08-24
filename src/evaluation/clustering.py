import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def cluster_embeddings(embeds, n_clusters, random_state=42):
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    return km.fit_predict(embeds.numpy())


def stable_ari_nmi(embeds, metadata, n_clusters=5, seeds=range(20), n_init=10):
    """ARI and NMI across many k-means seeds, not one.

    A single seed is not a measurement here. n_init keeps the best of its restarts by
    *inertia*, and a compact partition is not necessarily one that lines up with
    modality, so the same embeddings can score anywhere across a wide band depending
    on where k-means happened to start. Measured spreads reach 0.27 ARI on some models,
    which is far larger than any difference between models, so every single-seed number
    reported before this was noise dressed as a result.
    """
    modality = metadata['Modality'].values
    X = embeds.numpy()

    aris, nmis = [], []
    for seed in seeds:
        labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init=n_init).fit_predict(X)
        aris.append(adjusted_rand_score(modality, labels))
        nmis.append(normalized_mutual_info_score(modality, labels))

    aris, nmis = np.array(aris), np.array(nmis)
    return {
        'ari_mean': float(aris.mean()), 'ari_std': float(aris.std()),
        'ari_min': float(aris.min()), 'ari_max': float(aris.max()),
        'nmi_mean': float(nmis.mean()), 'nmi_std': float(nmis.std()),
        'n_seeds': len(aris),
    }


def evaluate_against_modality(cluster_labels, metadata):
    modality = metadata['Modality'].values
    ari = adjusted_rand_score(modality, cluster_labels)
    nmi = normalized_mutual_info_score(modality, cluster_labels)
    crosstab = pd.crosstab(pd.Series(cluster_labels, name='cluster'), pd.Series(modality, name='modality'))
    return ari, nmi, crosstab

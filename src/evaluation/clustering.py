import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def cluster_embeddings(embeds, n_clusters, random_state=42):
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    return km.fit_predict(embeds.numpy())


def evaluate_against_modality(cluster_labels, metadata):
    modality = metadata['Modality'].values
    ari = adjusted_rand_score(modality, cluster_labels)
    nmi = normalized_mutual_info_score(modality, cluster_labels)
    crosstab = pd.crosstab(pd.Series(cluster_labels, name='cluster'), pd.Series(modality, name='modality'))
    return ari, nmi, crosstab

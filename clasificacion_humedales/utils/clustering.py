import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from kneed import KneeLocator
from sklearn import mixture
from sklearn.decomposition import PCA

from clasificacion_humedales.utils.utils_maia import directories

PATH_IN, PATH_OUT = directories()


def _reduction_and_its_score(X, n):
    method = PCA(random_state=1, n_components=n, svd_solver='full')
    transformed = method.fit_transform(X)
    score = method.score(X)
    return {'transformed': transformed, 'score': score}


def _all_reductions_and_its_scores(X, n_components):
    reductions_and_its_scores = [_reduction_and_its_score(X, n) for n in n_components]
    return reductions_and_its_scores


def best_reduction(X, n_components):
    reductions_and_its_scores = _all_reductions_and_its_scores(X, n_components)
    scores = [reduction_and_its_score['score'] for reduction_and_its_score in reductions_and_its_scores]
    best_n = _best_n_components(n_components, scores)
    best_score_index = n_components.index(best_n)
    best_reduction = reductions_and_its_scores[best_score_index]['transformed']
    best_score = reductions_and_its_scores[best_score_index]['score']
    return {'best_reduction': best_reduction, 'scores': scores, 'best_score': best_score, 'best_n': best_n}


def _best_n_components(n_components, scores):
    return KneeLocator(n_components, scores, curve='concave', direction='increasing').knee


def graph_pca_results(n_components, result_pca, save_plot=False):
    title = 'PCA'
    n_components_pca = result_pca['best_n']
    pca_scores = result_pca['scores']

    figure = plt.figure()
    plt.plot(n_components, pca_scores, 'b', label='PCA scores')

    plt.axvline(n_components_pca, color='b',
                label='PCA Knee: %d' % n_components_pca, linestyle='--')
    plt.xticks(n_components)
    plt.xlabel('Cantidad de componentes principales')
    plt.ylabel('Scores')
    plt.legend(loc='lower right')
    plt.title(title)

    plt.show()
    if save_plot:
            figure.savefig(PATH_OUT + 'pca_results', bbox_inches='tight')


def _average_bic(n_clusters, X):
    sum_of_scores = 0
    seeds = 20
    for seed in range(seeds):
        gmm = mixture.GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=seed)
        gmm.fit_predict(X)
        sum_of_scores += gmm.bic(X)
    average = sum_of_scores / seeds
    return average


def clustering_with_best_bic(X, n_clusters_range):
    lowest_average_bic = np.infty
    average_bic_scores = []
    for n_clusters in n_clusters_range:
        average = _average_bic(n_clusters, X)
        average_bic_scores.append(average)
        if average_bic_scores[-1] < lowest_average_bic:
            lowest_average_bic = average_bic_scores[-1]
            best_n = n_clusters

    return {'lowest_average_bic': lowest_average_bic, 'best_n': best_n, 'average_bic_scores': average_bic_scores}


def clustering_with_best_seed(X, n, covariance_type='full'):
    best_bic_score = np.infty
    seeds = 20
    for seed in range(seeds):
        gmm = mixture.GaussianMixture(n_components=n, covariance_type=covariance_type, random_state=seed)
        clustered_image = gmm.fit_predict(X)
        bic_score = gmm.bic(X)
        if bic_score < best_bic_score:
            best_gmm = gmm
            best_cluster = clustered_image
            best_bic_score = bic_score
    return {'cluster': best_cluster, 'bic_score': best_bic_score}


def graph_bic_clustering(n_clusters_range, average_scores, name_graph):
    best_n = n_clusters_range[np.array(average_scores).argmin()]
    average_scores_dataframe = pd.DataFrame(
        data={'Cantidad de clusters': n_clusters_range, 'BIC promedio': average_scores})
    fig, axes = plt.subplots(1, 1, figsize=(10, 7))
    axes.set_xticks(n_clusters_range)
    plt.axvline(best_n, color='orange', label='K con mejor BIC promedio: %d' % best_n,
                linestyle='--')
    average_bic_graph = sns.scatterplot(data=average_scores_dataframe, x='Cantidad de clusters', y="BIC promedio")
    average_bic_graph.figure.savefig(PATH_OUT + name_graph)
    return average_bic_graph


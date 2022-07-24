from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from natsort import natsorted

from clasificacion_humedales.utils.utils_maia import directories, read_clustering

PATH_IN, PATH_OUT = directories()


def generate_clustering_history(label_history, folder):
    '''Busca en la carpeta el historial del agglomerative y devuelve una lista con todos esos clusterings.'''
    images_history = []
    for i in range(len(label_history)):
        image = read_clustering(PATH_OUT + '%s/iteracion_%s.tif' % (folder, i))
        image = np.array(image).astype(int)
        images_history.append(np.reshape(image, image.shape[0] * image.shape[1]))
    return images_history


def amount_of_classes_with_percentage_less_than(n, clustering):
    return (np.array(list(percentage_of_each_cluster(clustering).values()))<n).sum()


def first_iteration_without_small_clusters(clustering_history, boundary):
    for i, iteration in enumerate(clustering_history):
        if amount_of_classes_with_percentage_less_than(boundary, iteration)==0:
            return i, iteration


def compare_clusters(cover_map, clusters, measure, nan_value=-3000):
    return [measure(cluster[cover_map != nan_value], cover_map[cover_map != nan_value]) for cluster in clusters]


def amount_of_pixels_in_each_cluster(clustering):
    clusters, counts = np.unique(clustering, return_counts=True)
    pixels_per_cluster = dict(zip(clusters, counts))
    return pixels_per_cluster


def percentage_of_each_cluster(clustering):
    pixels_per_cluster = amount_of_pixels_in_each_cluster(clustering)
    if -3000 in pixels_per_cluster:
        del pixels_per_cluster[-3000]
    total_amount_of_pixels = sum(list(pixels_per_cluster.values()))
    for cluster in pixels_per_cluster.keys():
        pixels_per_cluster[cluster] = 100 * pixels_per_cluster[cluster] / total_amount_of_pixels
    return pixels_per_cluster


def pixels_cluster_1_in_each_class_of_clustering_2(cluster_1, clustering_1, clustering_2):
    '''Para cada clase encontrada en clustering_1 calcula en qué medida los pixels del mismo pertenecen a una clase dada del clustering_2
    '''
    indices_cluster_1 = np.where(clustering_1 == cluster_1)
    clusters, counts = np.unique(clustering_2[indices_cluster_1], return_counts=True)
    return dict(zip(clusters, counts))


def pixels_belonging(cluster_1, cluster_2, clustering_1, clustering_2):
    classes_and_amounts = pixels_cluster_1_in_each_class_of_clustering_2(cluster_1, clustering_1, clustering_2)
    return classes_and_amounts.get(cluster_2, 0)


def percentage_of_belonging(cluster_1, cluster_2, clustering_1, clustering_2):
    classes_and_amounts = pixels_cluster_1_in_each_class_of_clustering_2(cluster_1, clustering_1, clustering_2)
    pixels_cluster_1 = sum(classes_and_amounts.values())
    return 100 * (classes_and_amounts.get(cluster_2, 0) / pixels_cluster_1)


def belonging_dataframe_absolute_values(clustering_1, clustering_1_labels, clustering_1_title, clustering_2,
                                        clustering_2_labels, clustering_2_title):
    belonging_matrix = []
    for clustering_1_label in clustering_1_labels:
        for clustering_2_label in clustering_2_labels:
            belonging_matrix.append([clustering_1_label, clustering_2_label,
                                     pixels_belonging(clustering_1_label, clustering_2_label, clustering_1,
                                                      clustering_2)])

    long_format_data = pd.DataFrame(belonging_matrix, columns=[clustering_1_title, clustering_2_title, 'interseccion'])

    return long_format_data.pivot(index=clustering_1_title, columns=clustering_2_title, values='interseccion')


def belonging_dataframe_percentages(clustering_1, clustering_1_labels, clustering_1_title, clustering_2,
                                    clustering_2_labels, clustering_2_title):
    percentage_of_each_cluster_clustering_2 = percentage_of_each_cluster(clustering_2)
    percentage_of_each_cluster_clustering_1 = percentage_of_each_cluster(clustering_1)
    belonging_matrix = []
    for clustering_1_label in clustering_1_labels:
        for clustering_2_label in clustering_2_labels:
            belonging_matrix.append([
                f'{clustering_1_label} \n {percentage_of_each_cluster_clustering_1[clustering_1_label].round(3)}% ',
                f'{clustering_2_label}: {percentage_of_each_cluster_clustering_2[clustering_2_label].round(3)}%',
                percentage_of_belonging(clustering_1_label, clustering_2_label, clustering_1,
                                        clustering_2)])

    long_format_data = pd.DataFrame(belonging_matrix, columns=[clustering_1_title, clustering_2_title, 'interseccion'])

    return long_format_data.pivot(index=clustering_1_title, columns=clustering_2_title, values='interseccion')


def complete_and_order(belonging, clustering):
    for i in range(len(np.unique(clustering)) - 1):
        if not i in belonging:
            belonging[i] = 0
    belonging.pop(-3000, None)
    sorted_belonging = OrderedDict(sorted(belonging.items()))
    return sorted_belonging


def find_minority_classes_for(index, clustering, percentage_boundary=2):
    if index==0:
        return [cluster for cluster, percentage in percentage_of_each_cluster(clustering).items() if percentage<percentage_boundary]
    else:
        return [f'{cluster}: {percentage.round(3)}%' for cluster, percentage in percentage_of_each_cluster(clustering).items() if percentage<percentage_boundary]


def drop_classes(ordered_df, minority_classes, index):
    if index==0:
        return ordered_df.reset_index().drop(minority_classes).set_index(ordered_df.index.name)
    else:
        return ordered_df.drop(minority_classes, axis=1)


def trimmed_and_ordered(df, index, clustering, percentage_boundary=2):
    '''Dado un index (fila o columna) y un dataframe que representa el heatmap con la
    pertenencia de las clases elimina las clases minoritarias para un treshold dado y ordena.'''
    minority_classes = find_minority_classes_for(index, clustering, percentage_boundary)
    ordered_df = df.reindex(natsorted(df.index), columns=natsorted(df.columns))
    return drop_classes(ordered_df, minority_classes, index)


def plot_heatmap(dataframe, with_title=False, save_plot=False):
    clustering_1_title = dataframe.index.name
    clustering_2_title = dataframe.columns.name
    fig, axes = plt.subplots(1, 1, figsize=(20, 10))
    plt.cla()
    plot = sns.heatmap(dataframe, annot=True, fmt='.1f')
    if with_title:
        axes.set_title(f'Píxels de {clustering_1_title} en cada clase de {clustering_2_title} para cada clase de {clustering_1_title}')
    if save_plot:
        plot.figure.savefig(PATH_OUT + f'heatmaps/{clustering_2_title} vs {clustering_1_title}', bbox_inches='tight')
    return plot

# Por ahora no los estoy usando. Los dejo por las dudas. Borrar en versión final.

# def belonging_dataframe_proxy(clustering_1, clustering_1_labels, clustering_1_title, clustering_2, clustering_2_title,
#                               purity_of_classes, uh):
#     matrix = np.array([row_for_class_proxy(cluster, clustering_1, clustering_2, purity_of_classes) for cluster in
#                        clustering_1_labels])
#     row_indices = [f'{cluster}: {percentage.round(3)}%' for cluster, percentage in
#                    percentage_of_each_cluster(clustering_1).items()]
#     column_indices = [f'{cluster}: {percentage.round(3)}%' for cluster, percentage in
#                       percentage_of_each_cluster(uh).items()]
#     df = pd.DataFrame(matrix, index=row_indices, columns=column_indices)
#     df.index.name = clustering_1_title
#     df.columns.name = clustering_2_title
#     return df


# def row_for_class_proxy(cluster, clustering_1, clustering_2, purity_of_classes):
#     belonging = pixels_cluster_1_in_each_class_of_clustering_2(cluster, clustering_1, clustering_2)
#     percentage_belonging = list(complete_and_order(belonging, clustering_2).values()) / \
#                            amount_of_pixels_in_each_cluster(clustering_1)[cluster]
#     return np.matmul(percentage_belonging, purity_of_classes) * 100


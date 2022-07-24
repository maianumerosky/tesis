from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.spatial.distance as dist
from clasificacion_humedales.utils.utils_maia import directories, read_image, save_clustering
from scipy.cluster import hierarchy
import os

PATH_IN, PATH_OUT = directories()
PATH_OUT += 'agglomerative/'


def agglomerative(clustering, metric):
    history_of_labels = []
    initial_labels = np.unique(clustering['label'].tolist())
    initial_amount_clusters = len(initial_labels)
    amount_remaining_clusters = initial_amount_clusters
    while amount_remaining_clusters > 1:
        closest = _closest_clusters_and_their_distance(clustering, metric)
        labels_to_merge = closest['labels']
        distance_between_clusters = closest['distance']
        new_label = 2 * initial_amount_clusters - amount_remaining_clusters
        _merge(clustering, labels_to_merge, new_label)
        _record(history_of_labels, labels_to_merge, distance_between_clusters)
        amount_remaining_clusters = len(np.unique(clustering['label'].tolist()))
    return history_of_labels


def find_and_replace(new_image, old_label, new_label):
    new_image[new_image == old_label] = new_label


def rename(clustering, old_label, new_label):
    clustering.loc[clustering['label'] == old_label, 'label'] = new_label


def create_clustering_df(image, clustering):
    return pd.DataFrame({'NDVI': image.tolist(), 'label': clustering})


def _merge(clustering, labels_to_merge, new_label):
    rename(clustering, labels_to_merge[0], new_label)
    rename(clustering, labels_to_merge[1], new_label)


def _record(history_of_labels, labels_to_merge, distance_between_clusters):
    history_of_labels.append([labels_to_merge[0], labels_to_merge[1], distance_between_clusters])


def average_distance(cluster_1, cluster_2):
    distance_matrix = dist.cdist(cluster_1, cluster_2)
    return np.average(distance_matrix)


def maximum_distance(cluster_1, cluster_2):
    distance_matrix = dist.cdist(cluster_1, cluster_2)
    return np.amax(distance_matrix)


def ward_distance(cluster_1, cluster_2):
    center_cluster_1 = np.mean(cluster_1)
    center_cluster_2 = np.mean(cluster_2)
    c1 = len(cluster_1)
    c2 = len(cluster_2)
    return ((c1 * c2) / (c1 + c2)) * dist.euclidean(center_cluster_1, center_cluster_2)


def _closest_clusters_and_their_distance(clustering, metric):
    distance = np.infty
    labels_of_closest = ()
    labels = np.unique(clustering['label'].tolist())
    for label_1 in labels:
        for label_2 in labels:
            if label_1 != label_2:
                cluster_1 = _find_cluster_with(clustering, label_1)
                cluster_2 = _find_cluster_with(clustering, label_2)
                distance_between_clusters = metric(cluster_1, cluster_2)
                if distance_between_clusters < distance:
                    distance = distance_between_clusters
                    labels_of_closest = (label_1, label_2)
    return {'labels': labels_of_closest, 'distance': distance}


def _find_cluster_with(clustering, label):
    return clustering.query('label==%s' % label)['NDVI'].tolist()


def generate_images_history(original_clustering, label_history, template_path, folder_name):
    os.makedirs(PATH_OUT+folder_name)
    template_image = read_image(template_path)[0]
    dimensions = template_image.shape
    image = deepcopy(template_image)
    mask = image >= 0
    image[mask] = original_clustering
    image[np.invert(mask)] = -3000
    steps = len(label_history)
    amount_of_original_clusters = len(np.unique(original_clustering))-1
    for i in range(steps+1):
        new_label = amount_of_original_clusters + i
        if i!=0:            
            find_and_replace(image, label_history[i-1][0], new_label)
            find_and_replace(image, label_history[i-1][1], new_label)
        save_clustering(template_path, f'{PATH_OUT}{folder_name}/iteracion_{i}.tif', np.reshape(image, dimensions))


def _fake_amount_of_points(history):
    matrix = deepcopy(history)
    iterations = len(matrix)
    points_for_each_cluster = [1] * (iterations + 1)
    for i in range(iterations):
        label_1 = matrix[i][0]
        label_2 = matrix[i][1]
        amount_points_new_cluster = points_for_each_cluster[label_1] + points_for_each_cluster[label_2]
        points_for_each_cluster.append(amount_points_new_cluster)
        matrix[i].append(amount_points_new_cluster)
    return matrix, points_for_each_cluster


def create_dendrogram(clustering, plot_title):
    history, _ = _fake_amount_of_points(clustering)
    plt.figure(figsize=(30, 30))
    dendro = hierarchy.dendrogram(history, leaf_font_size=35)
    plt.savefig(f'{PATH_OUT}/{plot_title}.jpg')
    plt.show()
    return dendro
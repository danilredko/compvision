

import numpy as np

import matplotlib.pyplot as plt
import cv2
from scipy.spatial.distance import euclidean as euc
import random
from numpy.linalg import eig, svd, norm
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets.samples_generator import make_blobs
import os
import copy


class Node(object):
    def __init__(self, data):
        self.data = data
        self.children = []

    def add_child(self, obj):
        self.children.append(obj)


def count_nodes(tree):

    counter = 0

    if len(tree.children) != 0:

        for child in tree.children:

            counter = counter + count_nodes(child) + 1

    return counter

def print_tree(tree):

    if tree.children!=[]:

        print('HEAD:'+str(tree.data))

        for child in tree.children:
            print(child.data)

        print('********************')

        for child in tree.children:
            print_tree(child)
    else:

        print('HEAD:'+str(tree.data))
        print('*********************')


def construct_vocab_tree_helper(cluster, number_of_clusters, center,  model, max_depth, depth):

    tree = Node(center)
    data = Node(cluster)

    if max_depth > depth:

        model.fit(cluster)
        centers = model.cluster_centers_
        labels = model.labels_

        for i in range(number_of_clusters):

            new_cluster = cluster[np.where(labels == i, True, False)]

            child, data_child = construct_vocab_tree_helper(new_cluster, number_of_clusters, centers[i],  model, max_depth, depth+1)

            tree.add_child(child)
            data.add_child(data_child)

    return tree, data


def vocab_tree(features, branch_factor, max_depth):

    print('CONSTRUCTING VOCAB TREE')

    model1 = MiniBatchKMeans(n_clusters=1)
    model1.fit(features)
    first_center = model1.cluster_centers_

    model = MiniBatchKMeans(n_clusters=branch_factor)
    tree, data = construct_vocab_tree_helper(features, branch_factor, first_center, model, max_depth, 0)

    return tree, data


def find_leaf(tree, desc):

    if len(tree.children) == 0:
        return tree.data
    values = []
    for child in tree.children:
        score = euc(child.data, desc)
        values.append(score)
    ind_min = np.argmin(values)
    return find_leaf(tree.children[ind_min], desc)

num_of_features = 100*36
branch_factor = 3
max_depth = 3

A = np.random.randint(10, 1000, size=(num_of_features, 1))

tree, _ = vocab_tree(A.reshape(-1, 1), branch_factor, max_depth)

print_tree(tree)
from scipy.cluster.hierarchy import fclusterdata



import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.distance import euclidean as euc
import random
from numpy.linalg import eig, svd, norm


import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.distance import euclidean as euc
import random
from numpy.linalg import eig, svd, norm
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets.samples_generator import make_blobs
import os

def get_all_features(directory):

    '''

    Get all the features of the training images from our data using SIFT.

    '''

    f_list = os.listdir(directory)

    all_features = np.array([]).reshape(0, 128)

    for file in f_list:

        if not file.startswith('.') and not file.endswith('.gif'):

            image = cv2.imread('data/DVDcovers/'+str(file))[:, :, ::-1]

            sift = cv2.xfeatures2d.SIFT_create(nfeatures=500)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, descriptors = sift.detectAndCompute(gray, None)

            all_features = np.concatenate((all_features, descriptors), axis=0)

    return np.array(all_features)


class Node(object):
    def __init__(self, data):
        self.data = data
        self.children = []

    def add_child(self, obj):
        self.children.append(obj)


def tree_count(tree):

    counter = 0
    if tree.children!=[]:

        print('HEAD:'+str(tree.data))

        for child in tree.children:
            print(child.data)

        print('********************')

        counter += len(tree.children)

        for child in tree.children:
            counter += tree_count(child)
    else:

        print('HEAD:'+str(tree.data))
    return counter


def construct2d_tree(cluster, number_of_clusters, center, depth):

    tree = Node(center)
    data = Node(cluster)

    if max_depth > depth:

        model.fit(cluster)
        centers = model.cluster_centers_
        labels = model.labels_

        for i in range(number_of_clusters):

            new_cluster = cluster[np.where(labels == i, True, False)]

            child, data_child = construct2d_tree(new_cluster, number_of_clusters, centers[i], depth+1)

            tree.add_child(child)
            data.add_child(data_child)

    return tree, data

X, y = make_blobs(n_samples=10000, centers=1, n_features=2)
model1 = MiniBatchKMeans(n_clusters=1)
model1.fit(X)
first_center = model1.cluster_centers_

max_depth = 3
model = MiniBatchKMeans(n_clusters=3, max_iter=100)
tree_k, data_k = construct2d_tree(X, 3,  first_center, depth=0)
k = tree_count(tree_k)

print(k)


def my_plot(tree, data, max_depth):

    #Plot the data first

    layers_tree = []
    layers_data = []
    layers_data.append(data.children)
    layers_tree.append(tree.children)

    for j in range(max_depth):

        layer_tree_item = np.array([layers_tree[j][i].children for i in range(len(layers_tree[j]))]).flatten()
        layers_tree.append(layer_tree_item)
        layer_data_item = np.array([layers_data[j][i].children for i in range(len(layers_data[j]))]).flatten()
        layers_data.append(layer_data_item)

    plt.subplot(2, 2, 1)
    plt.scatter(data.data[:, [0]], data.data[:, [1]])
    plt.scatter(tree.data[0][0], tree.data[0][1], marker='^', color='black')
    plt.title('Iteration: 0')

    for num_layers in range(max_depth):

        plt.subplot(2, 2, num_layers+2)

        for data_item, tree_item in zip(layers_data[num_layers], layers_tree[num_layers]):
            plt.title('Iteration: {}'.format(num_layers+1))
            plt.scatter(data_item.data[:, [0]], data_item.data[:, [1]])
            plt.scatter(tree_item.data[0], tree_item.data[1], marker='^', color='black')
    plt.suptitle('K-Means for 2D')
    plt.show()


my_plot(tree_k, data_k, max_depth)

#features = get_all_features('data/DVDcovers')

#build_vocabulary_tree(features, 5, 3, 7)

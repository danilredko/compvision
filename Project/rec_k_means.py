
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import matplotlib.pyplot as plt


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
        print('*********************')
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
model = MiniBatchKMeans(n_clusters=5, max_iter=100)
tree_k, data_k = construct2d_tree(X, 5,  first_center, depth=0)
tree_count(tree_k)


def my_plot(tree, data, max_depth):

    layers_tree = []
    layers_data = []
    layers_data.append(data.children)
    layers_tree.append(tree.children)

    for j in range(max_depth):

        layer_tree_item = np.array([layers_tree[j][i].children for i in range(len(layers_tree[j]))]).flatten()
        layers_tree.append(layer_tree_item)
        layer_data_item = np.array([layers_data[j][i].children for i in range(len(layers_data[j]))]).flatten()
        layers_data.append(layer_data_item)

    for num_layers in range(max_depth):

        for data_item, tree_item in zip(layers_data[num_layers], layers_tree[num_layers]):
            plt.title('Iteration: {}'.format(num_layers+1))
            plt.scatter(data_item.data[:, [0]], data_item.data[:, [1]])
            plt.scatter(tree_item.data[0], tree_item.data[1], marker='^', color='black')

        plt.show()


my_plot(tree_k, data_k, 5)
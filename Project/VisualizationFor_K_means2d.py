
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean as euc


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


def construct2d_tree(cluster, number_of_clusters, center,  model, max_depth, depth):

    tree = Node(center)
    data = Node(cluster)

    if max_depth > depth:

        model.fit(cluster)
        centers = model.cluster_centers_
        labels = model.labels_

        for i in range(number_of_clusters):

            new_cluster = cluster[np.where(labels == i, True, False)]

            child, data_child = construct2d_tree(new_cluster, number_of_clusters, centers[i],  model, max_depth, depth+1)

            tree.add_child(child)
            data.add_child(data_child)

    return tree, data


def my_plot(tree, data, max_depth, branch_factor):

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
    plt.title('Iteration: 0 \n Branch Factor: {}'.format(branch_factor))

    for num_layers in range(max_depth):

        plt.subplot(2, 2, num_layers+2)

        for data_item, tree_item in zip(layers_data[num_layers], layers_tree[num_layers]):
            plt.title('Iteration: {} \n Branch Factor: {}'.format(num_layers+1, branch_factor))

            plt.scatter(data_item.data[:, [0]], data_item.data[:, [1]])
            plt.scatter(tree_item.data[0], tree_item.data[1], marker='^', color='black')
    plt.suptitle('Hierarchical K-Means for 2D')
    plt.show()


def main():
    branch_factor = 3

    X, y = make_blobs(n_samples=10000, centers=1, n_features=2)
    model1 = MiniBatchKMeans(n_clusters=1)
    model1.fit(X)
    first_center = model1.cluster_centers_

    max_depth = 3
    model = MiniBatchKMeans(n_clusters=branch_factor, max_iter=100)
    tree_k, data_k = construct2d_tree(X, branch_factor,  first_center,  model, max_depth, depth=0)

    layers_tree = []
    layers_data = []
    layers_data.append(data_k.children)
    layers_tree.append(tree_k.children)

    number_of_nodes = count_nodes(tree_k)
    print_tree(tree_k)
    print('Tree has {} nodes'.format(number_of_nodes))
    my_plot(tree_k, data_k, max_depth, branch_factor)

main()

'''
tree = Node(19)

c1 = Node(15)
c2 = Node(12)
c3 = Node(16)
c4 = Node(76)
tree.add_child(c1)
tree.add_child(c2)
tree.add_child(c3)
tree.add_child(c4)

c1.add_child(Node(10))
c1.add_child(Node(8))
c1.add_child(Node(7))
c1.add_child(Node(4))

c2.add_child(Node(6))
c2.add_child(Node(5))
c2.add_child(Node(2))
c2.add_child(Node(32))


c3.add_child(Node(7))
c3.add_child(Node(9))
c3.add_child(Node(1))
c3.add_child(Node(34))

c4.add_child(Node(1))
c4.add_child(Node(54))
c4.add_child(Node(66))
c4.add_child(Node(65))



def find_leaf(tree, desc):

    if len(tree.children)==0:

        return tree.data
    values = []
    for child in tree.children:
        score = euc(child.data, desc)
        values.append(score)
    ind_min = np.argmin(values)
    return find_leaf(tree.children[ind_min], desc)


def get_all_leaves(tree):

    if len(tree.children)==0:
        return tree.data

    leaves = []

    for child in tree.children:

        leaves.append(get_all_leaves(child))

    return np.array(leaves).flatten()




table = get_all_leaves(tree)

f = 'kekos'
l = 'asdfas'
d = np.array([1, 4, 12, 1])

print(table.reshape(4, 4))

table = np.array([[1,4,5,6], [1,4,12,1], [1234,2,1,4]])


zeros = [[]] * table.shape[0]


def add_word(word, table, f):

    ind_find = np.where(np.all(table==word,axis=1))[0][0]

    if zeros[ind_find]==[]:

        zeros[ind_find] = [f]
    else:
        #if f not in zeros[ind_find]:
            zeros[ind_find].append(f)

    return zeros

k  = add_word([1,4,5,6],table, 'fasdfadf')

k = add_word([1,4,5,6], table,'fasdfadf' )

print(k)



#A[[item_index], [1]] = ['here']


#print('LEAF :{}'.format(find_leaf(tree, 2)))
'''
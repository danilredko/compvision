

import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.distance import euclidean as euc
import random
from numpy.linalg import eig, svd, norm
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets.samples_generator import make_blobs
import os


def get_descriptors(image, n_features):

    sift = cv2.xfeatures2d.SIFT_create(nfeatures=n_features)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, descriptors = sift.detectAndCompute(gray, None)
    return descriptors


def get_all_features(directory):

    print('GETTING THE FEATURES')

    '''

    Get all the features of the training images from our data using SIFT.

    '''

    f_list = os.listdir(directory)

    file_dictionary = {}

    all_features = np.array([]).reshape(0, 128)
    #all_features = []

    for file in f_list:

        if not file.startswith('.') and not file.endswith('.gif'):

            image = cv2.imread('data/DVDcovers/'+str(file))[:, :, ::-1]

            descriptors = get_descriptors(image, 250)

            file_dictionary[file] = descriptors

            all_features = np.concatenate((all_features, descriptors), axis=0)
            #all_features.append(descriptors)

    print('DONE GETTING FEATURES')

    return np.array(all_features), file_dictionary


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


def vocab_tree(features, directory, branch_factor, max_depth):

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


def match_descriptor_query(image, tree, dict_of_words):

    # Returns a dictionary where key is the visual word, and the value is 1 every time a word is added

    descriptors = get_descriptors(image, 250)

    for desc in descriptors:

        word = find_leaf(tree, desc)

        dict_of_words[np.str(word)].append(1)

    return dict_of_words


def build_inverted_index(descriptors, vocab_tree, file_dictionary):

    invert_table = {}

    for file_name in file_dictionary.keys():

        for desc in file_dictionary[file_name]:

            word = find_leaf(vocab_tree, desc)

            if np.str(word) in invert_table.keys():
                invert_table[np.str(word)].append(file_name)

            else:

                invert_table[np.str(word)] = []

    return invert_table


def dict_print(dict):

    for item in dict.keys():
        print(dict[item])


def get_the_dictionary_of_words(dict):

    for item in dict.keys():

        dict[item] = []

    return dict

directory = 'data/DVDCovers'
branch = 9
max_depth = 3
D, file_dic = get_all_features(directory)  # is the set of all descriptors

tree, _ = vocab_tree(D, directory, branch, max_depth)

test_image = cv2.imread('data/test/image_01.jpeg')[:, :, ::-1]

'''
counter = 0
for item in query_image_matched.keys():
    counter = counter +1
    print(query_image_matched[item])
print(counter)
'''
inv = build_inverted_index(D, tree, file_dic)

words = get_the_dictionary_of_words(inv)

query_image_matched = match_descriptor_query(test_image, tree, words)

dict_print(query_image_matched)
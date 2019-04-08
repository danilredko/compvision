

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

            sift = cv2.xfeatures2d.SIFT_create(nfeatures=100)
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


def vocab_tree(directory, branch_factor, max_depth):

    features = get_all_features(directory)

    model1 = MiniBatchKMeans(n_clusters=1)
    model1.fit(features)
    first_center = model1.cluster_centers_

    model = MiniBatchKMeans(n_clusters=branch_factor)
    tree, data = construct_vocab_tree_helper(features, branch_factor, first_center, model, max_depth, 0)

    return tree, data


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


def create_inverted_index(directory, tree, branch_factor):

    table_of_words = get_all_leaves(tree)

    table_of_words = table_of_words.reshape(table_of_words.shape[0]/128, 128)

    # CHECK LATER

    file_names_list = [[]] * table_of_words.shape[0]

    f_list = os.listdir(directory)

    for file_name in f_list:

        if not file_name.startswith('.') and not file_name.endswith('.gif'):
            image = cv2.imread('data/DVDcovers/'+str(file_name))[:, :, ::-1]

            sift = cv2.xfeatures2d.SIFT_create(nfeatures=100)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, descriptors = sift.detectAndCompute(gray, None)

            for desc in descriptors:

                word = find_leaf(tree, desc)

                ind_find = np.where(np.all(table_of_words == word, axis=1))[0][0]

                # if file name is not in the list, then it's added the the
                if len(file_names_list[ind_find])==0:

                    file_names_list[ind_find] = [file_name]

                else:
                    # File name is not added to the list if it's already in the list
                    #if file_name not in file_names_list[ind_find]:

                        file_names_list[ind_find].append(file_name)

    return table_of_words, file_names_list


tree, data = vocab_tree('data/DVDcovers/', 5 , 3)

tablewords, filenamelist = create_inverted_index('data/DVDcovers', tree, 3)


def get_frequency_of_words(file_name_list, directory):

    f_list = os.listdir(directory)

    images_frequency = []

    for file_name in f_list:

        if not file_name.startswith('.') and not file_name.endswith('.gif'):

            image_i_frequency = []

            for word in file_name_list:
                number_times_word_i = word.count(file_name)
                image_i_frequency.append(number_times_word_i)

            images_frequency.append(np.array(image_i_frequency))

    return np.array(images_frequency)


frequencies = get_frequency_of_words(filenamelist, 'data/DVDcovers')


def get_hist_test_image(file_name, tree, table_of_words, list_of_files_names):

    image = cv2.imread('data/test/'+str(file_name))[:, :, ::-1]

    sift = cv2.xfeatures2d.SIFT_create(nfeatures=100)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, descriptors = sift.detectAndCompute(gray, None)
    possible_images = []

    for desc in descriptors[0:1]:

        word = find_leaf(tree, desc)

        ind_find = np.where(np.all(table_of_words == word, axis=1))[0][0]

        k = list_of_files_names[ind_find]

        print(k)

        print(np.unique(k).shape)

        possible_images.append(k)

    #possible_images = np.array(possible_images)

    return possible_images


kek = get_hist_test_image('image_01.jpeg', tree, tablewords, filenamelist)
'''
uniques_elements = []

for item in kek:

    uniques_elements.append(item)

print(np.unique(uniques_elements))

'''
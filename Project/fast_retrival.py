

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

    print('GETTING THE FEATURES')

    '''

    Get all the features of the training images from our data using SIFT.

    '''

    f_list = os.listdir(directory)

    all_features = np.array([]).reshape(0, 128)

    for file in f_list:

        if not file.startswith('.') and not file.endswith('.gif'):

            image = cv2.imread('data/DVDcovers/'+str(file))[:, :, ::-1]

            sift = cv2.xfeatures2d.SIFT_create(nfeatures=1000)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, descriptors = sift.detectAndCompute(gray, None)

            all_features = np.concatenate((all_features, descriptors), axis=0)

    print('DONE GETTING FEATURES')

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

    print('CONSTRUCTING VOCAB TREE')

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

    print('CREATING INVERTED INDEX FILE')

    table_of_words = get_all_leaves(tree)

    table_of_words = table_of_words.reshape(table_of_words.shape[0]/128, 128)

    print('Number of leaves: ' + str(table_of_words.shape[0]))

    # CHECK LATER

    file_names_list = [[]] * table_of_words.shape[0]

    f_list = os.listdir(directory)

    for file_name in f_list:

        if not file_name.startswith('.') and not file_name.endswith('.gif'):

            image = cv2.imread('data/DVDcovers/'+str(file_name))[:, :, ::-1]

            sift = cv2.xfeatures2d.SIFT_create(nfeatures=1000)
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
                    if file_name not in file_names_list[ind_find]:

                        file_names_list[ind_find].append(file_name)

    print('DONE CREATING INDEX FILE')
    return table_of_words, file_names_list


def get_frequency_of_words(file_name_list, directory):

    print('GETTING FREQUENCIES OF WORDS')

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


def get_n_i(filenamelists):

    n_i = []
    for item in filenamelist:
        n_i.append(len(item))
    return np.array(n_i)

def get_hist_test_image(file_name, tree, table_of_words, list_of_files_names):

    print('GETTING HISTOGRAM OF TEST IMAGE')

    image = cv2.imread('data/test/'+str(file_name))[:, :, ::-1]

    sift = cv2.xfeatures2d.SIFT_create(nfeatures=1000)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, descriptors = sift.detectAndCompute(gray, None)
    possible_images = []

    freq_word = [0] * table_of_words.shape[0]

    for desc in descriptors:

        word = find_leaf(tree, desc)

        ind_find = np.where(np.all(table_of_words == word, axis=1))[0][0]

        k = list_of_files_names[ind_find]

        freq_word[ind_find] = freq_word[ind_find] + 1

        possible_images.append(k)

    #possible_images = np.array(possible_images)

    #return np.array(possible_images)
    return possible_images, np.array(freq_word)

branch_factor = 10
max_depth = 3

tree, data = vocab_tree('data/DVDcovers', branch_factor, max_depth)

tablewords, filenamelist = create_inverted_index('data/DVDcovers', tree, branch_factor)

frequencies = get_frequency_of_words(filenamelist, 'data/DVDcovers')

pi, freq_w = get_hist_test_image('image_07.jpeg', tree, tablewords, filenamelist)

N = len(os.listdir('data/DVDcovers'))  # Number of documents

n_i = get_n_i(filenamelist)

n_d = np.sum(freq_w)  # total number of words in the image


def tf_idf(N, n_i, n_d, n_id):

    return np.true_divide(n_id, n_d) * np.log(np.true_divide(N, n_i))

t_test_image = tf_idf(N, n_i, n_d, freq_w)


norms = []

for item in frequencies:

    n_id = item

    n_d = np.sum(n_i)

    train_image = tf_idf(N, n_i, n_d, n_id)

    norm = np.dot(train_image, t_test_image) / (np.linalg.norm(train_image) * np.linalg.norm(t_test_image))

    norms.append(norm)

norms = np.array(norms)

top_index = norms.argsort()[-10:][::-1]

h = os.listdir('data/DVDcovers')

for i in top_index:
    print(h[i])


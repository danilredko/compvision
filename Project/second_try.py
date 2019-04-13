

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


invert_table = {}
query_table = {}
words = {}

counter_words = 0
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

            image = cv2.imread(directory+str(file))[:, :, ::-1]

            descriptors = get_descriptors(image, num_of_features)

            file_dictionary[file] = descriptors

            all_features = np.concatenate((all_features, descriptors), axis=0)

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


def count_leaves(dict):

    return len(dict.keys())


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


def get_words_dict(tree):

    global words
    global counter_words

    if len(tree.children) == 0:
        words['word'+str(counter_words)] = list(tree.data)
        invert_table['word'+str(counter_words)] = {}
        #query_table['word'+str(counter_words)] = {}
        counter_words += 1

    for child in tree.children:

        get_words_dict(child)


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


def match_descriptor_query(image, tree):

    print('START MATCHING THE QUERY IMAGE')

    dict_of_words = {}

    # Returns a dictionary where key is the visual word, and the value is 1 every time a word is added

    descriptors = get_descriptors(image, num_of_features)

    for desc in descriptors:

        word = find_leaf(tree, desc)

        if np.str(word) in dict_of_words.keys():

            dict_of_words[np.str(word)].append(1)

        else:

            dict_of_words[np.str(word)] = []

    print("DONE WITH")

    return dict_of_words


def build_inverted_index(file_name, tree):

    global invert_table

    descriptors = get_descriptors(cv2.imread(directory + file_name), num_of_features)

    for desc in descriptors:

        word = find_leaf(tree, desc)

        word_name = words.keys()[words.values().index(list(word))]

        if file_name in invert_table[word_name]:

            invert_table[word_name][file_name] += 1

        else:

            invert_table[word_name][file_name] = 1


def build_inverted_index_query(file_name, tree):

    global query_table

    descriptors = get_descriptors(cv2.imread(file_name), num_of_features)

    for desc in descriptors:

        word = find_leaf(tree, desc)

        word_name = words.keys()[words.values().index(list(word))]

        if word_name not in query_table.keys():
            query_table[word_name] = {}

        if file_name in query_table[word_name]:

            query_table[word_name][file_name] += 1

        else:

            query_table[word_name][file_name] = 1

    print('DONE BUILDING INVERTED INDEX FILE QUERY')


def dict_print(dict, unique=False):

    for item in dict.keys():
        print('KEY: '+str(item))
        if unique:
            print(np.unique(dict[item]))
        else:
            print(dict[item])

'''
def get_list_of_possible_images(inv, words):

    possible_images = []

    words_of_query_image = words.keys()

    for word in words_of_query_image:

        for file_name in inv[word]:

            if file_name not in possible_images:

                possible_images.append(file_name)

    return possible_images
'''


def weight(word):

    n_i = len(invert_table[word])

    if n_i == 0:
        print('ERROR')
        return 0

    return np.log(np.true_divide(N, n_i))


def get_word_frequency(word, image, query=False):



    dic = invert_table

    if image in dic[word].keys():

        n_id = dic[word][image]

    else:

        n_id = 0

    n_d = 0 # total number of words in the image

    for item in dic.values():

        if image not in item.keys():

           n_d += 0

        else:

            n_d += item[image]

    if n_d==0:

        print('Error')

    return np.true_divide(n_id, n_d)



def get_word_frequency_query(word, image):

    n_id = query_table[word][image]

    n_d = 0 # total number of words in the image

    for item in query_table.values():

        if image not in item.keys():

           n_d += 0

        else:

            n_d += item[image]

    if n_d==0:

        print('Error')

    return np.true_divide(n_id, n_d)


def tf_idf(image, query):

    t_i = []

    for word in words.keys():

        inverse_document_freq = weight(word)
        if query:

            if word not in query_table.keys():
                freq_w = 0
            else:
                freq_w = get_word_frequency_query(word, image)
        else:
            freq_w = get_word_frequency(word, image)

        t_i.append(inverse_document_freq * freq_w)

    return np.array(t_i)


def build_inv_file(directory, tree):

    print("STARTING BUILDING INVERTED INDEX FILE")

    f_list = os.listdir(directory)

    for file_name in f_list:

        if not file_name.startswith('.') and not file_name.endswith('.gif'):

            print(file_name)

            build_inverted_index(file_name, tree)
    print('END BUILDING INVERTED INDEX FILE')


def compute_tf_idf(directory):

    dict_of_tf_idf = {}

    f_list = os.listdir(directory)

    for file_name in f_list:

        if not file_name.startswith('.') and not file_name.endswith('.gif'):

            dict_of_tf_idf[file_name] = list(tf_idf(file_name, False))

    return dict_of_tf_idf


def get_best_match(dict, top_k, query_tf_idf_score):

    score = {}

    best_k_images = []

    for item in dict.values():

        s = np.dot(item, query_tf_idf_score) / (np.linalg.norm(item) * np.linalg.norm(query_tf_idf_score))

        score[s] = dict.keys()[dict.values().index(list(item))]

    top_scores = sorted(score.keys())[-top_k:]

    for value in top_scores:

        best_k_images.append(score[value])

    print(score)

    return best_k_images

directory = 'data/DVDCovers/'
branch = 3
max_depth = 4

num_of_features = 500
N = len(os.listdir(directory))

D, file_dic = get_all_features(directory)  # is the set of all descriptors
tree, _ = vocab_tree(D, directory, branch, max_depth)

get_words_dict(tree)
build_inv_file(directory, tree)

build_inverted_index_query('data/test/image_07.jpeg', tree)
d_td = compute_tf_idf(directory)
f = tf_idf('data/test/image_07.jpeg', True)

top_k_images = get_best_match(d_td, 10, f)

print(top_k_images)





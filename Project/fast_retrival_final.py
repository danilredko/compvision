

import numpy as np
import cv2
from scipy.spatial.distance import euclidean as euc
from sklearn.cluster import MiniBatchKMeans
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

    all_features = np.array([]).reshape(0, 128)

    for file in f_list:

        if not file.startswith('.') and not file.endswith('.gif'):

            image = cv2.imread(directory+str(file))[:, :, ::-1]

            descriptors = get_descriptors(image, num_of_features)

            all_features = np.concatenate((all_features, descriptors), axis=0)

    print('DONE GETTING FEATURES')

    return np.array(all_features)


class Node(object):
    def __init__(self, data):
        self.data = data
        self.children = []

    def add_child(self, obj):
        self.children.append(obj)


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


def get_words_dict(tree, flist):

    global words
    global counter_words

    if len(tree.children) == 0:
        words['word'+str(counter_words)] = list(tree.data)

        d = {}
        for f in flist:
            d[f] = 0
        invert_table['word'+str(counter_words)] = d
        counter_words += 1

    for child in tree.children:

        get_words_dict(child, flist)


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


def build_inverted_index(file_name, tree):

    global invert_table

    descriptors = get_descriptors(cv2.imread(directory + file_name), num_of_features)

    for desc in descriptors:

        word = find_leaf(tree, desc)

        word_name = words.keys()[words.values().index(list(word))]

        invert_table[word_name][file_name] += 1


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


def weight(word):

    n_i = np.sum(invert_table[word].values())

    return np.log(np.true_divide(N, n_i))


def get_word_frequency(word, image, n_d,  query=False):

    n_id = invert_table[word][image]

    return np.true_divide(n_id, n_d)


def get_word_frequency_query(word, image, n_d_query):

    n_id = query_table[word][image]

    return np.true_divide(n_id, n_d_query)


def tf_idf(image, query):

    t_i = []

    if query:

        n_d_query = 0

        for word in query_table.keys():

            if query_table[word][image]!=0:

                n_d_query += 1
    else:

        n_d = 0

        for word in query_table.keys():

            if invert_table[word][image]!=0:

                n_d += 1

    for word in query_table.keys():

        inverse_document_freq = weight(word)

        if query:

            freq_w = get_word_frequency_query(word, image, n_d_query)
        else:

            freq_w = get_word_frequency(word, image, n_d)

        t_i.append(inverse_document_freq * freq_w)

    return np.array(t_i)


def build_inv_file(directory, tree):

    print("STARTING BUILDING INVERTED INDEX FILE")

    f_list = os.listdir(directory)

    for file_name in f_list:

        if not file_name.startswith('.') and not file_name.endswith('.gif'):

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

    return best_k_images


def get_best_k_matches(directory, test_image_directory, branch_factor, max_depth, best_k):

    list_of_files = os.listdir(directory)

    D = get_all_features(directory)  # is the set of all descriptors
    tree, _ = vocab_tree(D, directory, branch_factor, max_depth)

    get_words_dict(tree, list_of_files)

    build_inv_file(directory, tree)

    build_inverted_index_query(test_image_directory, tree)

    tf_scores_of_images = compute_tf_idf(directory)
    
    test_image_tf_score = tf_idf(test_image_directory, True)

    top_k_images = get_best_match(tf_scores_of_images, best_k, test_image_tf_score)

    return top_k_images


directory='data/DVDCoversTrueData/'
invert_table = {}
query_table = {}
words = {}
counter_words = 0
num_of_features = 500
list_of_files = os.listdir(directory)
N = len(list_of_files) #Total Number of Documents

top10_matches = get_best_k_matches(directory,
                   test_image_directory='data/test/image_07.jpeg', branch_factor=5, max_depth=3, best_k=10)
print(top10_matches)
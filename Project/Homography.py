
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.distance import euclidean as euc
import random
from numpy.linalg import eig, norm

cover = cv2.imread('data/DVDcovers/matrix.jpg')[:, :, ::-1]
test = cv2.imread('data/test/image_07.jpeg')[:, :, ::-1]


def get_interest_points(img):

    sift = cv2.xfeatures2d.SIFT_create()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)
    return kp, des


def my_matching(img1, img2):
    
    kp1, des1 = get_interest_points(img1)
    kp2, des2 = get_interest_points(img2)
    match = []
    kp_diff = []

    for j in range(0, len(kp1)):
        for i in range(0, len(kp2)):
            euc_dist = euc(des1[j], des2[i])
            if euc_dist < 100:
                kp_diff.append(euc_dist)
                x_i, y_i = kp1[j].pt
                x_j, y_j = kp2[j].pt
                match.append([x_i, y_i, x_j, y_j])

    return np.array(match)

def matching(im1, im2):
    
    MAX_FEATURES = 500
    GOOD_MATCH_PERCENT = 0.15

    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt


    points1 = np.array(points1)
    points2 = np.array(points2)

    matches = np.concatenate((points1, points2), axis=1)

    return matches, points1, points2


def match_images(img1, img2, matches):
    
    img1_show = img1.copy()
    img2_show = img2.copy()

    f=plt.figure()
    for i in range(0, matches.shape[0]):

        x_i, y_i, x_j, y_j = matches[i]

        f.add_subplot(1, 2, 1)
        plt.scatter(x_i, y_i, color='r', linewidth=1)
        plt.imshow(img1_show)
        f.add_subplot(1, 2, 2)
        plt.imshow(img2_show)
        plt.scatter(x_j, y_j, color='b',  linewidth=1)

        plt.show()

def compute_homography(matches):

    #Create an array -  A

    A = []

    for i in range(0, matches.shape[0]):

        x_i, y_i, x_j, y_j = matches[i]

        row = [x_i, y_i, 1, 0, 0, 0, -x_j*x_i, -x_j*y_i, -x_j]

        A.append(row)
        
        row = [0, 0, 0, x_i, y_i, 1, -y_j*x_i, -y_j*y_i, -y_j]
        
        A.append(row)

    A = np.array(A)

    e_val, e_vec = eig(np.matmul(A.T, A))
   
    mh = e_vec[:, np.argmin(e_val)].reshape(3, 3)

    return mh

def match_point_homography(x, y, H):
    
    """
    point is a match of on the reference image
    
    H is a homography matrix (given by compute_homography function)
    
    returns a coordinates of the match on the test image
    
    """

    vector = np.array([x, y, 1])
    
    result = np.dot(H, vector) # Gets a vector of the form [ax', ay', a].T

    new_x, new_y , _ = result / result[-1]
    
    return new_x, new_y

def plot_rect(p1,p2,p3,p4,color,linewidth=2):

    plt.plot((p1[0], p2[0]), (p1[1], p2[1]), c=color, linewidth=linewidth)
    plt.plot((p1[0], p3[0]), (p1[1], p3[1]), c=color, linewidth=linewidth)
    plt.plot((p2[0], p4[0]), (p2[1], p4[1]), c=color, linewidth=linewidth)
    plt.plot((p3[0], p4[0]), (p3[1], p4[1]), c=color, linewidth=linewidth)

def transform_image(img1, img2, H):
    
    img1x, img1y, _ = img1.shape
    
    p1 = match_point_homography(0, 0 , H)
    p2 = match_point_homography(0.0, img1x, H)
    p3 = match_point_homography(img1y, 0.0, H)
    p4 = match_point_homography(img1y, img1x, H)

    plt.imshow(img2)
    plot_rect(p1, p2, p3, p4, 'r')
    #plt.show()



def RANSAC(matches):

    best_inliers = 0
    
    best_H  = None
    
    for i in range(0, 1000):
                                
        SampleMatches = np.array(random.sample(matches, 4))
        
        H = compute_homography(SampleMatches)
        
        inliers = 0 

        for match in matches:

            x_i, y_i, x_j, y_j = match

            new_x, new_y = match_point_homography(x_i, y_i, H)

            dist = norm(np.array([x_j - new_x, y_j - new_y]))

            if dist < 3:

                inliers += 1

        if inliers > best_inliers:
            
            best_inliers = inliers 

            best_H = H

    return best_H, best_inliers


def get_matrix_H(img1, img2):

    matches, points1, points2 = matching(img1, img2)

    #h, _ = cv2.findHomography(points1, points2, cv2.RANSAC)

    myH, number_of_inliers = RANSAC(matches)

    #p1, p2, p3, p4 = transform_image(cover, test, myH, False)

    return myH, number_of_inliers


    #transform_image(cover, test, h, False)





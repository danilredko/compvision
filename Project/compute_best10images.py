
from Homography import *


def compute_H_for_best10(images, test_image):

    test_image = cv2.imread('data/test/'+str(test_image))[:, :, ::-1]

    inliers = {}
    H_dict = {}

    for file_name in images:

        img = cv2.imread('data/DVDcovers/'+str(file_name))[:, :, ::-1]

        H, number_of_inliers = get_matrix_H(img, test_image)

        inliers[file_name] = number_of_inliers

        H_dict[file_name] = H

    return H_dict, inliers


def find_best_image(H_dict, inliers, test_image):

    test_image = cv2.imread('data/test/'+str(test_image))[:, :, ::-1]

    best_inlier = np.max(inliers.values())

    img_name = inliers.keys()[inliers.values().index(best_inlier)]

    myH = H_dict[img_name]

    image = cv2.imread('data/DVDcovers/'+str(img_name))[:, :, ::-1]

    transform_image(image, test_image, myH)
    plt.title('Computed Homography \n Number of inliers: {}'.format(best_inlier))
    plt.axis('off')
    plt.show()

    plt.imshow(image)
    plt.title('Retrieved Image DVD Cover')
    plt.axis('off')
    plt.show()


def run(names_of_best_images, name_of_test_image):

    dictH, dict_inliers = compute_H_for_best10(names_of_best_images, name_of_test_image)

    find_best_image(dictH, dict_inliers, name_of_test_image)

best_images = ['matrix.jpg', 'shall_we_dance.jpg' ,'shrek2.jpg', 'shrek_the_musical.jpg',
 'silver_linings_playbook.jpg' ,'six_days_seven_nights.jpg', 'slap_shot.jpg',
 'sleepless_in_seattle.jpg' ,'sommersby.jpg' ,'stand_by_me.jpg']

run(best_images, 'image_07.jpeg')

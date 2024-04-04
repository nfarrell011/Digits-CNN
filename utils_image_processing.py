"""
    Problem Set 5: Recognition Using Deep Networks
    Joseph Nelson Farrell & Harshil Bhojwani 
    5330 Computer Vision and Pattern Recognition
    Northeastern University
    Bruce Maxwell, PhD.
    4.1.2024
    
    This file contains a library of utility functions that will process and read in images.

    Functions List (in order):

        1. process_images
        2. get_processed_images
"""
# import libraries
import numpy as np
import cv2
import os

def process_images(raw_image_folder, curated_image_folder):
    """
        Function: process_images
            This function will iterate over a directory containing raw image files and generate a directory
            of processed images.

        Parameters:
            raw_image_folder: (str) - name of folder containing raw images.
            curated_image_folder: (str) - name of folder to store processed images.

        Returns: 
            None
    """
    
    # iterate over the image files in the image folder
    for file_name in os.listdir(raw_image_folder):

        # make sure it's an image
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):

            # set image path
            image_path = os.path.join(raw_image_folder, file_name)

            # read in the image
            image = cv2.imread(image_path)

            # convert to greyscale
            grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # resize image
            resized_image = cv2.resize(grey_image, (28, 28), interpolation = cv2.INTER_AREA)

            # invert the image colors
            binary_img = np.invert(resized_image)

            # rotate the image
            binary_img = cv2.rotate(binary_img, cv2.ROTATE_90_CLOCKWISE)

            # set output
            out = os.path.join(curated_image_folder, file_name)

            # save new image
            cv2.imwrite(out, binary_img)

    return None

def get_processed_images(curated_image_folder):
    """
        Function: get_processed_images
            This function will iterate over a directory containing processed images and generate a list of
            images and image labels.

        Parameters:
            curated_image_folder: (str) - name of folder to store processed images.

        Returns: 
            image_list: (list) - list of image data
            true_label_list: (list) - list of corresponding true labels.
    """ 
    # image list
    image_list = []
    true_label_list = []

    # iterate over the image files in the image folder
    for file_name in os.listdir(curated_image_folder):

        # make sure it's an image
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):

            # true label string
            label, _ = file_name.split('.')

            # capitalize_label
            label = label.capitalize()

            # set image path
            image_path = os.path.join(curated_image_folder, file_name)

            # read in the image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # update lists
            true_label_list.append(label)
            image_list.append(image)


    return image_list, true_label_list


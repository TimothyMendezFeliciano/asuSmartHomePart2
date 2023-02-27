# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:44:25 2021

@author: chakati
"""
import cv2
import numpy as np
import os
import tensorflow as tf
import frameextractor
import csv
import handshape_feature_extractor as handshape
from utils import last_file_in_folder, find_train_data_equivalent_key, find_comparable_vectors, \
    return_correct_label, define_train_gesture, define_test_gesture

## import the handfeature extractor class

# =============================================================================
# Get the penultimate layer for trainig data
# =============================================================================
# your code goes here
# Extract the middle frame of each gesture video

training_directory = 'traindata/'
train_middle_frames_directory = 'trainMiddleFrames/'
train_count = 0
train_vector_list = {}

handshape_obj = handshape.HandShapeFeatureExtractor
for filename in os.listdir(training_directory):
    file = os.path.join(training_directory, filename)
    frameextractor.frameExtractor(file, train_middle_frames_directory, train_count)
    gesture = define_train_gesture(filename, train_count)
    lastFile = last_file_in_folder(train_middle_frames_directory)
    image = cv2.imread(lastFile, cv2.IMREAD_GRAYSCALE)
    train_vector_list[gesture] = (handshape_obj.extract_feature(handshape_obj.get_instance(), image))
    train_count = train_count + 1

# =============================================================================
# Get the penultimate layer for test data
# =============================================================================
# your code goes here
# Extract the middle frame of each gesture video

test_directory = 'test/'
test_middle_frames_directory = 'testMiddleFrames/'
test_count = 0
test_vector_list = {}
for filename in os.listdir(test_directory):
    file = os.path.join(test_directory, filename)
    frameextractor.frameExtractor(file, test_middle_frames_directory, test_count)
    gesture = define_test_gesture(filename, test_count)
    lastFile = last_file_in_folder(test_middle_frames_directory)
    image = cv2.imread(lastFile, cv2.IMREAD_GRAYSCALE)
    test_vector_list[gesture] = (handshape_obj.extract_feature(handshape_obj.get_instance(), image))
    test_count = test_count + 1

# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================

with open('Results.csv', 'w', newline='') as results_file:
    headers = ['Output Label']
    file_writer = csv.DictWriter(results_file, fieldnames=headers)
    for key, value in test_vector_list.items():
        minimum_cosine_difference = 1
        correct_label = key
        trainDataEquivalentKey = find_train_data_equivalent_key(key)
        for compareKey, compareValue in train_vector_list.items():
            calculated_difference = tf.keras.losses.cosine_similarity(
                value, compareValue, axis=-1
            )
            if calculated_difference < minimum_cosine_difference:
                minimum_cosine_difference = calculated_difference
                correct_label = compareKey
        output_label = return_correct_label(correct_label)
        file_writer.writerow({
            'Output Label': output_label
        })

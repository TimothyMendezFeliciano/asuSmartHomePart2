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
from utils import define_gesture, last_file_in_folder, find_train_data_equivalent_key, find_comparable_vectors, \
    return_correct_label

## import the handfeature extractor class

# =============================================================================
# Get the penultimate layer for trainig data
# =============================================================================
# your code goes here
# Extract the middle frame of each gesture video

trainingDirectory = 'traindata/'
trainMiddleFramesDirectory = 'trainMiddleFrames/'
trainCount = 0
trainVectorList = {}

handshapeObj = handshape.HandShapeFeatureExtractor
for filename in os.listdir(trainingDirectory):
    file = os.path.join(trainingDirectory, filename)
    frameextractor.frameExtractor(file, trainingDirectory + trainMiddleFramesDirectory, trainCount)
    gesture = define_gesture(filename, trainCount)
    lastFile = last_file_in_folder(trainingDirectory + trainMiddleFramesDirectory)
    image = cv2.imread(lastFile, cv2.IMREAD_GRAYSCALE)
    trainVectorList[gesture] = (handshapeObj.extract_feature(handshapeObj.get_instance(), image))
    trainCount = trainCount + 1

# =============================================================================
# Get the penultimate layer for test data
# =============================================================================
# your code goes here
# Extract the middle frame of each gesture video

testDirectory = 'test/'
testMiddleFramesDirectory = 'testMiddleFrames/'
testCount = 0
testVectorList = {}
for filename in os.listdir(testDirectory):
    file = os.path.join(testDirectory, filename)
    frameextractor.frameExtractor(file, testDirectory + testMiddleFramesDirectory, testCount)
    gesture = define_gesture(filename, testCount)
    lastFile = last_file_in_folder(testDirectory + testMiddleFramesDirectory)
    image = cv2.imread(lastFile, cv2.IMREAD_GRAYSCALE)
    testVectorList[gesture] = (handshapeObj.extract_feature(handshapeObj.get_instance(), image))
    testCount = testCount + 1

# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================

with open('Results.csv', 'w', newline='') as results_file:
    headers = ['Output_Label']
    file_writer = csv.DictWriter(results_file, fieldnames=headers)
    file_writer.writeheader()
    for key, value in testVectorList.items():
        minimum_cosine_difference = 1
        correct_label = key
        trainDataEquivalentKey = find_train_data_equivalent_key(key)
        for compareKey, compareValue in find_comparable_vectors(trainDataEquivalentKey, trainVectorList).items():
            calculated_difference = tf.keras.losses.cosine_similarity(
                value, compareValue, axis=-1
            )
            if calculated_difference < minimum_cosine_difference:
                minimum_cosine_difference = calculated_difference
                correct_label = compareKey
        # print("Splice/Split label", correct_label)
        correct_label = return_correct_label(correct_label)
        file_writer.writerow({
            'Output_Label': correct_label
        })

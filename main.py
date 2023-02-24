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

## import the handfeature extractor class

# =============================================================================
# Get the penultimate layer for trainig data
# =============================================================================
# your code goes here
# Extract the middle frame of each gesture video

trainingDirectory = 'traindata'
trainMiddleFramesDirectory = 'trainMiddleFrames'
testDirectory = 'test'
testMiddleFramesDirectory = 'testMiddleFrames'
testCount = 0
trainCount = 0
trainVectorList = []
testVectorList = []

handshapeObj = handshape.HandShapeFeatureExtractor
for filename in os.listdir(trainingDirectory):
    file = os.path.join(trainingDirectory, filename)
    frameextractor.frameExtractor(file, trainMiddleFramesDirectory, trainCount)
    trainCount = trainCount + 1

for filename in os.listdir(trainMiddleFramesDirectory):
    file = os.path.join(trainMiddleFramesDirectory, filename)
    image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    trainVectorList.append(handshapeObj.extract_feature(handshapeObj.get_instance(), image))

# =============================================================================
# Get the penultimate layer for test data
# =============================================================================
# your code goes here 
# Extract the middle frame of each gesture video
# for filename in os.listdir(testDirectory):
#     file = os.path.join(testDirectory, filename)
#     frameextractor.frameExtractor(file, testMiddleFramesDirectory, testCount)
#     testCount = testCount + 1

# for filename in os.listdir(testMiddleFramesDirectory):
#     file = os.path.join(testMiddleFramesDirectory, filename)
#     image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
#     testVectorList.append(handshapeObj.extract_feature(handshapeObj.get_instance(), image))

# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================

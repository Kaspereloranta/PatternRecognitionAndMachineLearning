# Kasper Eloranta, H274212, kasper.eloranta@tuni.fi
# Pattern Recognition and Machine Learning, DATA.ML.200
# Exercise 5

import pickle
import numpy as np
import csv
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import statistics
import os
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

def convertY(Y):
    Yc = np.zeros((len(Y),2))
    for index, value in enumerate(Y):
        Yc[int(index)][int(value-1)] = 1
    return Yc

def readImgs(path):
    list = []
    for root, _, files in os.walk(path):
        current_directory_path = os.path.abspath(root)
        for f in files:
            name, ext = os.path.splitext(f)
            if ext == ".jpg":
                current_image_path = os.path.join(current_directory_path, f)
                current_image = mpimg.imread(current_image_path)
                list.append(current_image)
    return np.array(list)

def class_acc(pred, gt):
    correctlyClassified = 0
    j = 0
    while j < len(pred):
        if pred[j] == gt[j]:
            correctlyClassified += 1
        j += 1
    accuracy = correctlyClassified / len(gt)
    return accuracy

def main():
    # loading data
    path1 = "GTSRB_subset_2/class1"
    path2 = "GTSRB_subset_2/class2"
    imgs1 = readImgs(path1)
    imgs2 = readImgs(path2)
    imgs1num = imgs1.shape[0]
    imgs2num = imgs2.shape[0]

    # Task 1
    trainPortion = 0.8
    imgs1trainX, imgs1testX = train_test_split(imgs1,test_size=1-trainPortion,random_state=25)
    imgs2trainX, imgs2testX = train_test_split(imgs2,test_size=1-trainPortion,random_state=25)

    imgs1trainY = np.ones(imgs1trainX.shape[0])
    imgs1testY = np.ones(imgs1testX.shape[0])
    imgs2trainY = np.full(imgs2trainX.shape[0],2)
    imgs2testY = np.full(imgs2testX.shape[0],2)

    trainX = np.concatenate((imgs1trainX,imgs2trainX))
    trainY = np.concatenate((imgs1trainY,imgs2trainY))
    testX = np.concatenate((imgs1testX,imgs2testX))
    testY = np.concatenate((imgs1testY,imgs2testY))
    trainY = convertY(trainY)
    testY = convertY(testY)
    trainX = trainX / 255
    testX = testX / 255

    # Task 2
    nnModel = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(64,64,3)),
        tf.keras.layers.Dense(100,activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(2, activation='sigmoid'),
    ])

    nnModel.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy'])
    nnModel.fit(trainX,trainY,epochs=10,verbose=2)

    nn_test_loss, nn_test_acc = nnModel.evaluate(testX, testY, verbose=2)
    print(nnModel.summary())

    print("Test set accuracy is:", nn_test_acc)

main()
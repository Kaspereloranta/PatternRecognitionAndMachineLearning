# Kasper Eloranta, H274212, kasper.eloranta@tuni.fi
# Pattern Recognition and Machine Learning, DATA.ML.200
# Exercise 4

import pickle
import numpy as np
import csv
import math
import matplotlib.pyplot as plt
import statistics
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression


def mapY(y):
    for index, data in enumerate(y):
        if (data == -1):
            y[index] = 0
    return y

# Loading data and normalizing data
x = np.loadtxt("X.dat", unpack=True)
y = np.loadtxt("y.dat", unpack=True)
y = mapY(y)
x[0] = x[0] - np.mean(x[0])
x[1] = x[1] - np.mean(x[1])
x = np.swapaxes(x, 0, 1)

def class_acc(pred, gt):
    correctlyClassified = 0
    j = 0
    while j < len(pred):
        if pred[j] == gt[j]:
            correctlyClassified += 1
        j += 1
    accuracy = correctlyClassified / len(gt)
    return accuracy

def predict(w, x):
    y = 1 / (1 + math.exp((w[0] * x[0] + w[1] * x[1])*-1))
    y = round(y)
    return y

def SSEgradient(w):
    w1 = 0
    w2 = 0
    for index, sample in enumerate(x):
        e = math.exp((sample[0] * w[0] + sample[1] * w[1])*-1)
        w1 = w1+(2*y[index]*e*sample[0])/(pow(1+e,2))-(2*sample[0]*(e+pow(e,2)))/(pow(1+2*e+pow(e,2),2))
        w2 = w2+(2*y[index]*e*sample[1])/(pow(1+e,2))-(2*sample[1]*(e+pow(e,2)))/(pow(1+2*e+pow(e,2),2))
    gradient = np.array((-w1, -w2))
    return gradient

def SSE():
    weights = np.empty((101, 2))
    weights[0] = (1, -1)
    mu = 0.001
    for index, wt in enumerate(weights):
        if (index == 100):
            continue
        else:
            weights[index + 1] = wt - mu * SSEgradient(wt)
    return weights


def MLgradient(w):
    w1 = 0
    w2 = 0
    for index, sample in enumerate(x):
        e = math.exp((sample[0] * w[0] + sample[1] * w[1])*-1)
        w1 = w1 + ((1-1/(1+e))*sample[0]-sample[0]/(1+e))
        w2 = w2 + ((1-1/(1+e))*sample[1]-sample[1]/(1+e))
    gradient = np.array((w1, w2))
    return gradient

def ML():
    weights = np.empty((101, 2))
    weights[0] = (1, -1)
    mu = 0.001
    for index, wt in enumerate(weights):
        if (index == 100):
            continue
        else:
            weights[index + 1] = wt + mu * MLgradient(wt)
    return weights

def main():
    # Task 1
    clf = LogisticRegression()
    clfPen = LogisticRegression(penalty="none", fit_intercept=False)
    clf.fit(x, y)
    clfPen.fit(x, y)
    accuracy = class_acc(clf.predict(x), y)
    accuracyPen = class_acc(clfPen.predict(x), y)
    SKaccuraciesPercent = np.ones(101) * accuracyPen * 100

    # Task 2
    weight_trajectory_SSE = SSE()
    SSEacc = np.empty((101, 1))
    for ind, w in enumerate(weight_trajectory_SSE):
        SSEpredictions = np.zeros(400)
        for index, pred in enumerate(SSEpredictions):
            SSEpredictions[index] = predict(w, x[index])
        SSEacc[ind] = class_acc(SSEpredictions, y) * 100

    # Task 3
    weight_trajectory_ML = ML()
    MLacc = np.empty((101, 1))
    testpred = np.zeros(400)
    for ind, w in enumerate(weight_trajectory_ML):
        MLpredictions = np.zeros(400)
        for index, pred in enumerate(MLpredictions):
            MLpredictions[index] = predict(w, x[index])
            if(ind==100):
                testpred[index] = predict(np.array((0.75,2)),x[index])
        MLacc[ind] = class_acc(MLpredictions, y) * 100
    print(class_acc(testpred,y)*100)

    # Plotting results
    plot_weight_SSE = np.swapaxes(weight_trajectory_SSE, 0, 1)
    plt.plot(plot_weight_SSE[0], plot_weight_SSE[1], 'r')
    plot_weight_ML = np.swapaxes(weight_trajectory_ML, 0, 1)
    plt.plot(plot_weight_ML[0], plot_weight_ML[1], 'b')
    plt.title("Optimization path")
    # plt.xlim([-1, 1.5])
    # plt.ylim([-1, 4])
    plt.xlabel("w1")
    plt.ylabel("w2")
    plt.legend(['SSE','ML'])
    plt.show()

    iterations = np.linspace(1, 101, 101)
    plt.plot(iterations, SKaccuraciesPercent, 'r--')
    plt.plot(iterations, SSEacc, 'b')
    plt.plot(iterations, MLacc, 'g')
    plt.title("Accuracies")
    plt.ylabel("Accuracy / %")
    plt.xlabel("Iteration")
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.legend(['SKLearn', 'SE rule classification accuracy',
                'ML rule classification accuracy'])
    plt.show()

main()
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def getSynTrainData() :
    data = np.genfromtxt(r"synthetic\train.txt", delimiter=",")
    return data


def getSynDevData() :
    data = np.genfromtxt(r"synthetic\dev.txt", delimiter=",")
    return data


def KMeansTrain(K, X) :
    num_samples = X.shape[0]
    num_features = X.shape[1]

    centroids = np.array([]).reshape(num_features, 0)
    for k in range(K):
        centroids = np.c_[centroids, X[random.randint(0, num_samples-1)]]

    for i in range(10):
        distances = np.array([]).reshape(num_samples, 0)
        for k in range(K):
            distances = np.c_[distances, np.sum((X - centroids[:,k])**2, axis=1)]

        cluster_numbers = np.argmin(distances, axis=1)
        
        clusters = {}
        for k in range(K):
            clusters[k] = np.array([]).reshape(0, num_features)

        for n in range(num_samples):
            clusters[cluster_numbers[n]] = np.r_["0,2", clusters[cluster_numbers[n]], X[n]]

        for k in range(K):
            centroids[:,k] = np.mean(clusters[k], axis=0)
    
    return centroids


def KMeansTest(X, centroids) : 
    num_samples = X.shape[0]
    K = centroids.shape[1]
    distances = np.array([]).reshape(num_samples, 0)
    for k in range(K) : 
        distances = np.c_[distances, np.sum((X - centroids[:,k])**2, axis=1)]
    
    cluster_numbers = np.argmin(distances, axis=1)
    return cluster_numbers 


if __name__ == "__main__" :
    syn_input = getSynTrainData()
    syn_data = np.array(syn_input[:,:2], dtype="float")
    syn_data1 = np.array(syn_data[:1250,:])
    syn_data2 = np.array(syn_data[1250:,:])
    syn_class = np.array(syn_input[:,2], dtype="int")

    K = 10
    syn_centroids1 = KMeansTrain(K, syn_data1)
    syn_centroids2 = KMeansTrain(K, syn_data2)
    syn_centroids = np.c_[syn_centroids1, syn_centroids2]

    syn_test_input = getSynDevData()
    syn_test1 = np.array(syn_test_input[:500,:2], dtype="float")
    syn_test2 = np.array(syn_test_input[500:,:2], dtype="float")

    true = 0
    cluster_numbers1 = KMeansTest(syn_test1, syn_centroids)
    cluster_numbers2 = KMeansTest(syn_test2, syn_centroids)

    for i in range(500) : 
        if(cluster_numbers1[i] < K) :
            true += 1
        if(cluster_numbers2[i] >= K) :
            true += 1

    print(true)
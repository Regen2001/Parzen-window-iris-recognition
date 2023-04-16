##
# @file     MQDF.py
#
# @date     2023-04
#
# @brief    Python code for INT301 Lab. Discriminant Functions & Non-parametric Classifiers
#           This code will implement the MQDF algorithm for iris.data classification
#           without using any third-party algorithm library.

# ----------------------------------------------------------------------------------------------------------- #
###############################################################################################################
#                             You need to fill the missing part of the code                                   #
#                        detailed instructions have been given in each function                              #
###############################################################################################################
# ----------------------------------------------------------------------------------------------------------- #

import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import timeit
from math import *
from Data_process import Data_process
import logging
import os
from pathlib import Path
from tqdm import tqdm

###############################################################################################################
#                                                  log                                                        #
###############################################################################################################

def create_log():
    log_dir = Path('./log/')
    log_dir.mkdir(exist_ok=True)
    logger = logging.getLogger("Parzen")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s_log.txt' % (log_dir, "Parzen"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def log_string(logger, str):
    logger.info(str)
    print(str)

###############################################################################################################
#                                                  draw                                                       #
###############################################################################################################

def draw(x, y, xlabel, ylabel, fileName, max=None):
    x = np.asarray(x)
    x = x.reshape(x.shape[0], -1)
    y = np.asarray(y)
    y = y.reshape(x.shape[0], -1)
    plt.figure(facecolor='w',edgecolor='w')
    plt.plot(x, y, linestyle = '-', linewidth = '1.5')
    plt.xlabel(xlabel, fontsize='x-large')
    plt.ylabel(ylabel, fontsize='x-large')
    plt.grid()
    if max != None:
            plt.annotate('Optimal Kernel Size, (%.1f, %.1f)' %(max[0], max[1]),
                 xy=(max[0], max[1]),
                 xycoords='data',
                 xytext=(-30, 30),
                 textcoords='offset points',
                 arrowprops=dict(arrowstyle="->"))
    plt.savefig('./log/'+fileName+'.png', dpi=600, format='png')
    temp_array = np.append(x, y, axis=1)
    np.savetxt('./log/'+fileName+'.txt', temp_array, fmt='%2f', delimiter =",")

###############################################################################################################
#                                        Self-defined functions                                               #
###############################################################################################################

def multivariate_normal_kernel(x, xi, h, cov): # Kernel function for datasets higher than 1D
    ###############################################################################################################
    #                                   YOU NEED FILL FOLLOWING CODES:
    det_cov = np.linalg.det(cov) # Determinant of the covariance matrix. Tip: use np.linalg.det()
    inv_cov = np.linalg.inv(cov) # Inverse of the covariance matrix. Tip: np.linalg.inv()
    u = (x - xi) / h # Compute the distance, math is (x - xi)/h. You may need to turn it into matrix
    numer = np.exp(-1*u@inv_cov@u.T/2)  # Numerator of the multivariate gaussian distribution, math is pow(e, ( -0.5 * u * inv_cov * u.T))
    denom = np.power(np.power(2*np.pi, len(x)) * det_cov, 1/2) # Denominator of the multivariate gaussian distribution, math is pow(pow(2*pi, len(x)) * det_cov, 1/2)
    kernel = numer / denom
    ###############################################################################################################
    return kernel

def normal_kernel(x, xi, h): # Kernal function for 1D datasets
    ###############################################################################################################
    #                                   YOU NEED FILL FOLLOWING CODES:
    u = (x - xi) / h # Compute the distance, math is (x - xi) / h
    kernel = np.exp(-1 * u**2 / 2) / (np.sqrt(2) * np.pi) # math is :exp(-(abs(u) ** 2) / 2) / (sqrt(2 * pi)), use numpy methods will be helpful
    ###############################################################################################################
    return kernel

def parzen_window(test, train, h, d): # Parzen Window function to output the conditional probability
    ###############################################################################################################
    #                                   YOU NEED FILL FOLLOWING CODES:
    cov = np.identity(d) # Create an identity matrix scaled with the dimension of the dataset for the covariance matrix. Tip: use np.identity()
    ###############################################################################################################
    if d == 1: # For 1D, apply the normal distribution
        p = normal_kernel(test, train, h) / h
        return p
    else: # For higher dimension, apply the multivariate normal distribution
        p = multivariate_normal_kernel(test, train, h, cov) / np.power(h, d)
        return p

def pz_predict(x_len, np_array): # To find the predicted class of the test set
    x_pred = []
    for i in range(x_len):
        max = np.max(np_array[:,i]) # Get the maximum probability in the ith column (ith data of x)
        if max == np_array[0][i]: # If 'max' is equal to the ith value in setosa array
            pred = 'Iris-setosa'
        elif max == np_array[1][i]: # If 'max' is equal to the ith value in versicolor array
            pred = 'Iris-versicolor'
        else: #  If 'max' is equal to the ith value in virginica array
            pred = 'Iris-virginica'
        x_pred.append(pred) # Store the predicted class in the order of the test datasets
    return x_pred


def pz_accuracy(pred_class, class_x): # To obtain the accuracy of the predicted result
    acc = 0  # Initialize the accuracy
    for ind, pred in enumerate(pred_class):
        if pred == class_x[ind]: # Compare the predicted classes with the actual classes of the test set
            acc += 1 # Increase the accuracy parameter if it is correct
        else:
            pass # If not correct, pass
    return (acc / len(pred_class) * 100)

###############################################################################################################
#                                              Main Part                                                      #
###############################################################################################################

if __name__ == '__main__':
    logger = create_log()
    log_string(logger, 'starting...')
    iris = Data_process() # Define Class Data_process()
    irist_data = iris.load_data() # Load the iris dataset
    div_data = iris.numeric_n_name(irist_data) # Separate numeric dataset and class names
    iris.data_analyzer(div_data[0]) # Check the general properties of the dataset
    init_data = iris.shuffle() # Shuffle the dataset
    five_data = iris.separate_data()

    # Initialize lists for checking the results
    h_list = []
    Lh_list = []
    acc_list = []

    start = timeit.default_timer() # Start timer to count the running time of Parzen Window Method
    # for h in range(1, 4): # 'for loop' condition to compare the program time of Pazen Window and MQDF
    for h in np.arange(0.2, 3, 0.1): # Find the optimal kernel via Changing the h value
        opt_size = 0  # To find the optimal kernel size
        sum_avg_acc = 0 # To calculate the average accuracy of 5-fold cross validation
        for index in range(len(five_data)): # 5-fold Cross-Validation
            total_subset = iris.combine_train(index, five_data) # Index denotes the array for testing
            sep_dataset = iris.separate_class(total_subset[0]) # Return separated train datasets by three classes
            sep_data = [sep_dataset[0], sep_dataset[1], sep_dataset[2]]
            prior_prob = iris.prior_prob(sep_dataset) # Calculate the prior probabilties of three classes
            # Convert the three train datasets into numpy array
            np_se = np.array(iris.numeric_n_name(sep_data[0])[0])
            np_ver = np.array(iris.numeric_n_name(sep_data[1])[0])
            np_vir = np.array(iris.numeric_n_name(sep_data[2])[0])
            # Prepare the train dataset in 'float' type
            train = [np_se.astype(float), np_ver.astype(float), np_vir.astype(float)]

            d = len(np_se[0]) # Dimension of the dataset

            x = np.array(iris.numeric_n_name(total_subset[1])[0]) # Extract the numeric data of test set
            np_x = x.astype(float)
            x_len = len(np_x)
            class_x = iris.numeric_n_name(total_subset[1])[1] # Class names of each test data
            # To store the conditional probability of each test data
            p_se = []
            p_ver = []
            p_vir = []
            cn = 0 # Counter to check the category of the train dataset in for loop

            # Start the Parzen Window algorithm
            for name in train: # For three class names
                for x in np_x: # For each data list of the test set
                    p_x = 0 # define the initial probability of x
                    for x_i in name: # For each data list of the train set
                        con_prob = parzen_window(x, x_i, h, d) # Compute the kernel function of PZ
                        p_x += con_prob # Add the output of the kernel function for every train data lists
                    p_xw = p_x / len(name) # Compute the conditional probability of a test data
                    opt_size += log10(p_xw) # Maximum log-likelihood estimation to find the optimal kernel size

                    # Add the probability into its category
                    if cn == 0:
                        p_se.append(p_xw * prior_prob[0]) # Posterior probability of x in Setosa
                    elif cn == 1:
                        p_ver.append(p_xw * prior_prob[1]) # Posterior probability of x in Versicolor
                    else:
                        p_vir.append(p_xw * prior_prob[2]) # Posterior probability of x in Virginica
                cn += 1 # Count when the loop of one category is finished

            prob_array = np.array([p_se, p_ver, p_vir]) # Combine the computed posterior probability of three classes

            pred_class = pz_predict(x_len, prob_array) # Obtain the predicted results of Parzen Window Method

            pz_acc = pz_accuracy(pred_class, class_x) # Calculate the classification accuracy
            sum_avg_acc += pz_acc
            # print("Accuracy:", pz_acc)

        avg_acc = sum_avg_acc / len(five_data)  # Average accuracy

        Lh = opt_size / len(five_data) # Compute the average log-likelihood value of the chosen h

        log_string(logger, 'Average accuracy when h = %.1f is %.2f' %(h, avg_acc/100))

        # Store the results to plot on graphs
        h_list.append(h)
        Lh_list.append(Lh)
        acc_list.append(avg_acc)

    stop = timeit.default_timer() # Stop timer for the running time of Parzen Window algortihm
    # print('Running time of Parzen Window with 5-fold cross validation:', stop - start)
    log_string(logger, 'Running time of Parzen Window with 5-fold cross validation for each h: %.3f' % ((stop - start)/len(h_list)))

    # Plot the result of Maximum likelihood estimation of h & average classification accuracy in terms of h
    x = np.asarray(h_list)
    y = np.asarray(acc_list)
    y2 = np.asarray(Lh_list)

    # plot1 = plt.figure(1)
    # plt.plot(x, y, label="Accuracy")
    # plt.xlabel("h")
    # plt.ylabel("Average accuracy")

    draw(x, y, 'h', 'Average accuracy / %', 'par_accuracy')

    # plot2 = plt.figure(2)
    # plt.plot(x, y2, label="L(h)")
    # plt.xlabel("h")
    # plt.ylabel("L(h)")


    ymax = float(max(y2))
    xpos = np.where(y2 == ymax)
    xmax = float(x[xpos])
    opt_avg_acc = float(y[xpos])

    draw(x, y2, 'h', 'L(h)', 'L_H', [xmax, ymax])

    # plt.annotate('Optimal Kernel Size, (%.1f, %.1f)' %(xmax, ymax),
    #              xy=(xmax, ymax),
    #              xycoords='data',
    #              xytext=(-30, 30),
    #              textcoords='offset points',
    #              arrowprops=dict(arrowstyle="->"))
    # plt.legend()
    # plt.show()

    # print('Optimal Kernel Size:', xmax)
    log_string(logger, 'Optimal Kernel Size: %.1f' %(xmax))
    # print('Average accuracy when h =', xmax, ':', opt_avg_acc, '%')
    log_string(logger, 'The max average accuracy when h = %.1f is %.2f' %(xmax, opt_avg_acc/100))
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from math import *

###############################################################################################################
#                                        Self-defined functions                                               #
###############################################################################################################

def twoD_plot(filename):  # To check the general properties of the dataset in 2D (Additional task)
    data = pd.read_csv(filename, names=["sepal length", "sepal width", "petal length", "petal width", "class"])
    data.head(5)
    data.describe()
    data.groupby('class').size()
    sns.pairplot(data, hue="class", height=2, palette='colorblind');
    plt.show()


def fourD_plot(p1, p2, p3, p4):  # To check the general properties of the dataset in 4D (Additional task)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = p1.astype(float)  # sepal length
    y = p2.astype(float)  # sepal width
    z = p3.astype(float)  # petal length
    c = p4.astype(float)  # petal width
    img = ax.scatter(x, y, z, c=c,
                     cmap=plt.hot())  # The 4D datasets will be shown in the 3D coordinate with color gradient
    fig.colorbar(img)
    # Add axis
    ax.set_xlabel('sepal length', fontweight='bold')
    ax.set_ylabel('sepal width', fontweight='bold')
    ax.set_zlabel('petal length', fontweight='bold')
    plt.show()

###############################################################################################################
#                                   Class for Data pre-processing                                             #
###############################################################################################################

class Data_process:  # Class for data pre-processing
    def __init__(self):
        self.filename = "data/iris.data"  # Dataset folder name
        # Predefined parameters
        self.line_comp = []
        self.iris_list = []

    def load_data(self):  # Method to load the dataset and store them in a list
        with open(self.filename) as f:
            for line in f:
                text_lines = line.strip()
                line_comp = text_lines.split(',')
                self.iris_list.append(line_comp)
        del self.iris_list[-1]  # Remove the empty element of the list
        return self.iris_list

    def shuffle(self):  # Method to shuffle the stored dataset
        random.seed(17)  # Define the seed value first to keep the shuffled data same
        random.shuffle(self.iris_list)  # Shuffle the list
        return self.iris_list

    def separate_data(self):  # Method to separate the dataset into five parts for 5-fold cross validation
        length = int(len(self.iris_list) / 5)  # Cutting length of the list
        data1 = self.iris_list[:length]
        data2 = self.iris_list[length:length * 2]
        data3 = self.iris_list[length * 2:length * 3]
        data4 = self.iris_list[length * 3:length * 4]
        data5 = self.iris_list[length * 4:length * 5]
        return data1, data2, data3, data4, data5

    def combine_train(self, ind, total_data):  # Method to separate combined train sets and a test set
        train = []
        for i in range(len(total_data)):  # According to the index, the test set will be chosen among the five subsets
            if ind == i:
                test = total_data[i]
            else:
                train += total_data[i]
        return train, test

    def separate_class(self, dataset):  # Method to separate dataset into three given classes
        setosa = []
        versicolor = []
        virginica = []
        for info in dataset:
            if info[4] == 'Iris-setosa':
                setosa.append(info)
            elif info[4] == 'Iris-versicolor':
                versicolor.append(info)
            else:
                virginica.append(info)
        return setosa, versicolor, virginica

    def numeric_n_name(self, nested_list):  # Method to separate the numeric data and class_names
        num_list = []
        class_list = []
        for instance in nested_list:
            num_data = instance[:4]  # Extract the numeric data
            class_name = instance[4:]  # Extract the class names of the data sets
            num_list.append(num_data)
            class_list += class_name
        return num_list, class_list  # Numeric data can be converted into numpy array

    def data_analyzer(self, info):  # Method to plot the 2D and 4D figures of the given dataset to analyze the properties
        np_info = np.array(info)
        sepal_length = np_info[:, 0]
        sepal_width = np_info[:, 1]
        petal_length = np_info[:, 2]
        petal_width = np_info[:, 3]

        fourD_plot(sepal_length, sepal_width, petal_length, petal_width)
        twoD_plot(self.filename)

    def prior_prob(self, dataset):  # Method to calculate the prior probabilities of each class
        prior_prob_se = len(dataset[0]) / (len(dataset[0]) + len(dataset[1]) + len(dataset[2]))  # Setosa
        prior_prob_ve = len(dataset[1]) / (len(dataset[0]) + len(dataset[1]) + len(dataset[2]))  # Versicolor
        prior_prob_vi = len(dataset[2]) / (len(dataset[0]) + len(dataset[1]) + len(dataset[2]))  # Virginica
        return prior_prob_se, prior_prob_ve, prior_prob_vi
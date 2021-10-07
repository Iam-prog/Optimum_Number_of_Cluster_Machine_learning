# Finding the optimum number of clusters for the given dataset using Elbow Method.

import math
import sys
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Clustering:
    @property
    def datasetName(self):
        return self.datasetName

    @datasetName.setter
    def datasetName(self, datasetName):
        self.datasetName = datasetName

    @property
    def sk(self):
        return self.sk

    @sk.setter
    def sk(self, sk_value):
        self.sk = sk_value

    @property
    def ek(self):
        return self.ek

    @ek.setter
    def ek(self, ek_value):
        self.ek = ek_value

    @property
    def x(self):
        return self.x

    @x.setter
    def x(self, x):
        self.x = x

    @property
    def y(self):
        return self.y

    @y.setter
    def y(self, y):
        self.y = y

    @property
    def xLevel(self):
        return self.xLevel

    @xLevel.setter
    def xLevel(self, xLevel):
        self.outputLevel = xLevel

    @property
    def outputLevel(self):
        return self.outputLevel

    @outputLevel.setter
    def outputLevel(self, outputLevel):
        self.outputLevel = outputLevel

    # This function sets the default value
    def set_default_value(a):
        Clustering.datasetName = "Clustering.csv"
        Clustering.sk = "1"
        Clustering.ek = "10"
        Clustering.xLevel = ""
        Clustering.outputLevel = ""

    # This function reads the given switch value
    def read_switch(NumOfParams):
        for i in range(1, NumOfParams):
            if sys.argv[i].replace(" ", "") == '-d':
                Clustering.datasetName = sys.argv[i + 1]
            elif sys.argv[i].replace(" ", "") == '-sk':
                Clustering.sk = sys.argv[i + 1]
            elif sys.argv[i].replace(" ", "") == '-ek':
                Clustering.ek = sys.argv[i + 1]
            elif sys.argv[i].replace(" ", "") == '-xl':
                Clustering.xLevel = sys.argv[i + 1]
            elif sys.argv[i].replace(" ", "") == '-ol':
                Clustering.outputLevel = sys.argv[i + 1]

    # This function reads the given dataset
    def read_dataset(datasetName):
        dataset = pd.read_csv(datasetName)
        Clustering.dataset_split(dataset, Clustering.xLevel, Clustering.outputLevel)
        return dataset

    # This function splits x when Columns name given
    def x_split_with_class(dataset, xclassLevel):
        x = dataset[xclassLevel]
        x = x.to_frame()
        x.columns = ["X"]
        Clustering.x = x

    # This function splits x when no Columns name given
    def x_split_without_class(dataset):
        x = dataset[dataset.columns[0]]
        x = x.to_frame()
        x.columns = ["X"]
        Clustering.x = x

    # This function splits y when Columns name given
    def y_split_with_class(dataset, classLevel):
        y = dataset[classLevel]
        y = y.to_frame()
        y.columns = ["Y"]
        Clustering.y = y

    # This function splits y when no Columns name given
    def y_split_without_class(dataset):
        y = dataset.iloc[:, -1]
        y = y.to_frame()
        y.columns = ["Y"]
        Clustering.y = y

    # This function splits the target and X
    def dataset_split(dataset, xclassLevel, classLevel):
        if len(classLevel) != 0 and len(xclassLevel) != 0:
            Clustering.y_split_with_class(dataset, classLevel)
            Clustering.x_split_with_class(dataset, xclassLevel)
        elif len(classLevel) == 0 and len(xclassLevel) != 0:
            Clustering.y_split_without_class(dataset)
            Clustering.x_split_with_class(dataset, xclassLevel)
        elif len(classLevel) != 0 and len(xclassLevel) == 0:
            Clustering.y_split_with_class(dataset, classLevel)
            Clustering.x_split_without_class(dataset)
        else:
            Clustering.y_split_without_class(dataset)
            Clustering.x_split_without_class(dataset)

    # This function select the Cluster
    def select(x, y, n):
        selected = []
        for i in range(n):
            temp = [x[i], y[i]]
            selected.append(temp)
        return selected

    # This function Calculates the distance between two points
    def distance_calculation(a, b):
        return math.sqrt(math.pow((a[0]-b[0]),2)+(math.pow((a[1]-b[1]),2)))

    # This function Calculates the distance between the Cluster and all points
    def all_distance(selected_Cluster_points, all_points):
        all_calculated_distance = []
        for i in range(len(selected_Cluster_points)):
            temp = []
            for j in range(len(all_points)):
                value = Clustering.distance_calculation(selected_Cluster_points[i], all_points[j])
                temp.append(value)
            all_calculated_distance.append(temp)
            print("\n--> Distance from (C", i + 1, ") <--")
            print(temp)
        return all_calculated_distance

    # This function Calculates the Index number of minimum distance between the Cluster
    def cluster_points_minimum_distance_index_number(all_Calculated_Distance):
        minimum_distance_index_number = []
        for i in range(len(all_Calculated_Distance[0])):
            temp = []
            for j in range(len(all_Calculated_Distance)):
                temp.append(all_Calculated_Distance[j][i])
            result = np.where(temp == np.amin(temp))
            result = np.asarray(result)
            minimum_distance_index_number.append(result[0][0])
        print("\n\nIndex number of minimum distance between the Cluster: ")
        print(minimum_distance_index_number)
        return minimum_distance_index_number

    # This function gets the Cluster points
    def cluster_points(all_points, minimum_Distance_index_number, i):
        temp1 = []
        for j in range(len(all_points)):
            if minimum_Distance_index_number[j] == i:
                temp1.append(all_points[j])
        print("\nCluster ", i + 1, "( C", i + 1, ") Points: ")
        print(temp1)
        return temp1

    # This function Selected the New Centroid points
    def new_selected_centroid_points(selected_Centroid, all_points, minimum_Distance_index_number):
        temp = []
        new_selected_centroid_points = []
        print("\n\n---> Using Minimum Distance. New Cluster Points. <---")
        for i in range(len(selected_Centroid)):
            temp1 = Clustering.cluster_points(all_points, minimum_Distance_index_number, i)
            temp3 = []
            for k in range(len(temp1[0])):
                sum = 0
                for l in range(len(temp1)):
                    sum += temp1[l][k]
                temp2 = sum / len(temp1)
                temp3.append(temp2)
            new_selected_centroid_points.append(temp3)
            temp.append(temp1)
        print("\n\n---> New Cluster Centroid Points <---")
        print(new_selected_centroid_points)
        return new_selected_centroid_points

    # This function Calculates the Total Cost for All Cluster Points
    def cost(selected_Centroid, all_points, minimum_Distance_index_number, all_Calculated_Distance):
        sums = 0
        for i in range(len(selected_Centroid)):
            Clustering.cluster_points(all_points, minimum_Distance_index_number, i)
            sum_value = 0
            for k in range(len(minimum_Distance_index_number)):
                if minimum_Distance_index_number[k] == i:
                    sum_value = sum_value + all_Calculated_Distance[i][k]
            print("Cost of this Cluster = ", sum_value)
            sums = sums + sum_value
        print("\nTotal Cost = ", sums)
        return sums

    # This function plots the data
    def visualization(x, y, k_value, all_new_cost):
        plt.subplot(1, 2, 1)
        plt.scatter(x, y, color='red')
        plt.title('Scatter plot between X and Y')
        plt.xlabel("X")
        plt.ylabel("Y")

        plt.subplot(1, 2, 2)
        plt.scatter(k_value, all_new_cost, color='red')
        plt.plot(k_value, all_new_cost, c='red', lw=2)
        plt.xticks(np.arange(int(Clustering.sk), int(Clustering.ek) + 1, 1))
        plt.title('Value of K vs Squared Error (Cost)')
        plt.xlabel('Value of K')
        plt.ylabel('Squared Error (Cost)')

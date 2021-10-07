# Finding the optimum number of clusters for the given dataset using Elbow Method.

#           Instruction

# Use Switch ( -d )  for Dataset
# Use Switch ( -sk ) to start Number of K
# Use Switch ( -sk ) to end Number of K
# Use Switch ( -xl ) for xLevel; like what is X point
# Use Switch ( -ol ) for OutputLevel; like y or target or y point
# Example of an input given below -->
# py Clustering.py -d Clustering.csv -sk 2 -ek 10 -xl X -ol Y

import sys
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator

import Class

if __name__ == '__main__':
    NumOfParams = len(sys.argv)
    print("\nNumber of Parameter is : ", NumOfParams)

    Class.Clustering.set_default_value(0)
    Class.Clustering.read_switch(NumOfParams)

    print("Dataset Name is             (-b) : ", Class.Clustering.datasetName)
    print("Starting K value           (-sk) : ", Class.Clustering.sk)
    print("Ending K value             (-ek) : ", Class.Clustering.ek)
    print("X Level is                 (-xl) : ", Class.Clustering.xLevel)
    print("OutputLevel is             (-ol) : ", Class.Clustering.outputLevel)

    dataset = Class.Clustering.read_dataset(Class.Clustering.datasetName)

    print("\n-> Given X values <- \n", Class.Clustering.x)
    print("\n-> Given Y values <- \n", Class.Clustering.y)

    x = (Class.Clustering.x.to_numpy()).T.flatten()
    y = (Class.Clustering.y.to_numpy()).T.flatten()

    if len(x) < int(Class.Clustering.ek) or int(Class.Clustering.ek) < 3:
        print("\n**************************************** Warning ****************************************\n")
        print("***  Value of Ending K is higher than X size or less then 3 which is not acceptable.  ***")
        print("***          Highest Ending K acceptable for this dataset is : ", len(x), "                   ***")
        print("***                           So, taking highest Ending k value                       ***")
        print("\n**************************************** Warning ****************************************\n")
        Class.Clustering.ek = str(len(x))

    if len(x) <= int(Class.Clustering.sk) or int(Class.Clustering.sk) < 1:
        print("\n************************************* Warning *************************************\n")
        print("***       Value of Starting K is higher or equal to X size or less then 1.      ***")
        print("***     Which is not possible. So, taking default Starting k value 1.           ***")
        print("\n************************************* Warning *************************************\n")
        Class.Clustering.sk = str(1)

    all_points = Class.Clustering.select(x, y, len(x))
    print("\n---> So, All 2D points <---")
    print(all_points)

    all_cost = [0] * (int(Class.Clustering.sk))
    all_new_cost =[]

    print("\n\n---> Algorithm -> K-Means <---")

    for n in range(int(Class.Clustering.sk), int(Class.Clustering.ek) + 1):
        print("\n\n\n*********************************************")
        print("\n******* When the value of K is: ", n, "  *******")
        print("\n*********************************************")

        minimum_Distance_index_number = np.arange(0, len(all_points))

        selected_Centroid = Class.Clustering.select(x, y, n)
        print("\n")
        print("---> Input Cluster Centroid Points <---")
        print(selected_Centroid)

        count = 1
        while True:
            previous_minimum_Distance_index_number = minimum_Distance_index_number
            print("\n\n\n********* Iterations Number : ", count, "*********")
            flag = True
            all_Calculated_Distance = Class.Clustering.all_distance(selected_Centroid, all_points)
            minimum_Distance_index_number = Class.Clustering.cluster_points_minimum_distance_index_number(
                all_Calculated_Distance)

            for i in range(len(minimum_Distance_index_number)):
                if minimum_Distance_index_number[i] != previous_minimum_Distance_index_number[i]:
                    flag = False

            if flag:
                print("\n---> As Current Cluster Centroid points are the same as previous Cluster Centroid points. We "
                      "can stop the process now. <---")
                break

            selected_Centroid = Class.Clustering.new_selected_centroid_points(selected_Centroid, all_points,
                                                                              minimum_Distance_index_number)
            count += 1

        print("\n\n--> So, The Final Cluster Centroid Points <---")
        print(selected_Centroid)

        print("\n---> So, The Final Cluster Points with the Cost <---")
        cost = Class.Clustering.cost(selected_Centroid, all_points, minimum_Distance_index_number,
                                     all_Calculated_Distance)
        all_cost.append(cost)
        all_new_cost.append(cost)

    print(all_cost)

    print("\n\n---> All Costs based on the K values <---\n")
    for i in range(int(Class.Clustering.sk) , int(Class.Clustering.ek) + 1):
        print("Cost for k", i," is ->", all_cost[i])

    k_value = np.arange(int(Class.Clustering.sk) , int(Class.Clustering.ek) + 1)

    Class.Clustering.visualization(x, y, k_value, all_new_cost)

    try:
        optimum_number_of_cluster = KneeLocator(k_value, all_new_cost, curve='convex', direction='decreasing')
        optimum_number_of_cluster_value = optimum_number_of_cluster.knee
        plt.vlines(optimum_number_of_cluster_value, plt.ylim()[0], plt.ylim()[1], linestyles='dashed', colors='black')
        print("\nAs we can see from the Graph.")
        print("The Optimum number of Clusters for the given dataset using"
              " Elbow Method is: ", optimum_number_of_cluster_value)
    except:
        print("\nAn exception occurred.")
        print("As we can see from the Graph. There is no specific bend (knee / elbow) found in the plot.")
        print("So, we can not get the 'Optimum number' of Clusters for the given dataset using Elbow Method for "
              "these k values.")

    print("\n--> Visualization <--\n")
    plt.show()

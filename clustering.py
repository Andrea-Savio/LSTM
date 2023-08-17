#!/usr/bin/env python3

import rospy
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score                         
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import euclidean
from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseWithCovariance
#from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from spencer_tracking_msgs.msg import DetectedPersons
from spencer_tracking_msgs.msg import DetectedPerson

# Global variable to store received position data
position_data = []

# Function to calculate the Dunn Index
def dunn_index(X, labels):
    # Calculate the distance between cluster centers
    cluster_centers = []
    for cluster_label in np.unique(labels):
        cluster_points = X[labels == cluster_label]
        cluster_center = np.mean(cluster_points, axis=0)
        cluster_centers.append(cluster_center)
    
    inter_cluster_distances = pairwise_distances(cluster_centers, metric='euclidean')
    min_inter_cluster_distance = np.min(inter_cluster_distances[inter_cluster_distances > 0])
    
    # Calculate the maximum diameter of individual clusters
    max_intra_cluster_diameter = 0
    for cluster_label in np.unique(labels):
        cluster_points = X[labels == cluster_label]
        if len(cluster_points) > 1:
            cluster_diameter = max(euclidean(p1, p2) for i, p1 in enumerate(cluster_points) for p2 in cluster_points[i+1:])
            if cluster_diameter > max_intra_cluster_diameter:
                max_intra_cluster_diameter = cluster_diameter
    
    # Calculate the Dunn Index
    dunn_index = min_inter_cluster_distance / max_intra_cluster_diameter
    return dunn_index

# Callback function to process ROS message containing position data
def position_callback(data):
    #global position_data
    rospy.loginfo("inside")
    for item in data.detections:
        print(item)
        x = item.pose.pose.position.x
        y = item.pose.pose.position.y
        position_data.append([x, y])
    
    #rospy.loginfo(position_data)
    perform_clustering()    

# Function to perform clustering using DBSCAN and calculate silhouette score
def perform_clustering():
    global position_data

    if len(position_data) == 0:
        print("No data received for clustering.")
        return

    # Convert position_data to a numpy array for clustering
    position_array = np.array(position_data)

    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=2, min_samples=5)
    labels = dbscan.fit_predict(position_array)

    # Calculate various clustering evaluation metrics
    silhouette = silhouette_score(position_array, labels)
    davies_bouldin = davies_bouldin_score(position_array, labels)
    calinski_harabasz = calinski_harabasz_score(position_array, labels)
    dunn = dunn_index(position_array, labels)

    # Process the clustering results
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # Exclude outliers (-1)
    print(f"Number of Clusters: {num_clusters}")
    print(f"Silhouette Score: {silhouette}")
    print("Davies-Bouldin Index:", davies_bouldin)
    print("Calinski-Harabasz Index:", calinski_harabasz)
    print("Dunn Index:", dunn)

    # Plot

    for i in range(num_clusters):
        cluster_points = position_array[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i+1}")

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Clustering Results')
    plt.legend()
    plt.grid()
    plt.show()

# Initialize ROS node and subscribe to the position topic
def main():
    rospy.init_node('position_clustering_node')
    rospy.Subscriber('/spencer/perception_internal/detected_persons/laser_front_high_recall', DetectedPersons, position_callback)
    # Wait for some time to collect enough position data before clustering
    rospy.sleep(5)

    # Perform clustering after collecting enough data
    #perform_clustering()

    rospy.spin()

if __name__ == '__main__':
    main()
    
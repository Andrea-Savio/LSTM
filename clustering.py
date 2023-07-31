#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseWithCovariance
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from spencer_tracking_msgs.msg import DetectedPersons
from spencer_tracking_msgs.msg import DetectedPerson


# Global variable to store received position data
position_data = []

# Callback function to process ROS message containing position data
def position_callback(data):
    #global position_data
    rospy.loginfo("inside")
    for item in data.detections:
        print(item)
        x = item.pose.position.x
        y = item.pose.position.y
        position_data.append([x, y])
    
    rospy.loginfo(position_data)
    perform_clustering()    
    
def perform_clustering():
    # Convert position_data to a numpy array for clustering
    position_array = np.array(position_data)

    # Perform K-Means clustering (you can change the number of clusters as per your requirement)
    num_clusters = 3
    kmeans = KMeans(n_clusters=num_clusters)
    labels = kmeans.fit_predict(position_array)
    rospy.loginfo(position_array)
    # Plot the clusters
    plt.figure()
    for i in range(num_clusters):
        cluster_points = position_array[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i+1}")

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Clustering Results')
    plt.legend()
    plt.grid()
    plt.show()

    if __name__ == '__main__':
        rospy.init_node('position_clustering_node')
        rospy.Subscriber('/spencer/perception_internal/detected_persons/laser_front_high_recall', DetectedPersons, position_callback)

        # Wait for some time to collect enough position data before clustering
        #rospy.sleep(5)

        # Perform clustering after collecting enough data
        #perform_clustering()

        rospy.spin()
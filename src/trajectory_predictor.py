#!/usr/bin/env python3

import rospy
import csv
import sys
import matplotlib.pyplot as plt
import itertools
import time
from lstm_trainer2 import LSTM_Trainer
import torch
import torch.nn
import numpy as np
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Pose, PoseArray
from spencer_tracking_msgs.msg import TrackedPersons
from spencer_tracking_msgs.msg import TrackedPerson
from trajectory_prediction.msg import PredictedTrajectories
from trajectory_prediction.msg import PredictedTrajectory
from pickle import load

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score                         
from sklearn.metrics import pairwise_distances
from clustering import dunn_index

from sklearn.preprocessing import MinMaxScaler, StandardScaler


#msg = None

# Class for saving data of tracked people to use for prediction

class Tracked():
  def __init__(self, id, seq):
    self.id = id
    cluster = None
    #self.path = [[0]*3 for i in range(seq)]
    self.path = []
    self.context = []
    self.counter = 0
    self.w = 0
    self.seq = seq
  
  def get_path(self):
    return self.path
  
  def get_id(self):
    return id

  def get_counter(self, num):
    if self.counter == num:
      return True
    else:
      return False

  def add_detection(self, x, y, rot):#, z):
    self.w = rot
    if len(self.path) < self.seq:
      #column = [0,0,0]
      column = [0,0]
      column[0] = x
      column[1] = y
      #column[2] = z
      self.path.append(column)
      self.counter = self.counter + 1
    else:
      self.path.pop(0)
      self.counter = self.counter - 1
      self.add_detection(x, y)#, z)  

#------------------------------------------------------------------------------------------------------------------------------------

# Subscriber for receiving data of tracked people
"""

class TrackerListener():
  def __init__(self):
    #rospy.init_node('tracker_subscriber')
    self.msg = None
    self.sub = rospy.Subscriber("/spencer/perception/tracked_persons", TrackedPersons, self.tracker_callback)

  def tracker_callback(self, data):
    rospy.loginfo("Returning tracked people data")
    self.msg = data

  def get_data(self):
    return self.msg
"""
#------------------------------------------------------------------------------------------------------------------------------------  

"""
def trajectory_publisher(traj):
  
  rospy.init_node("trajectory_publisher")

  pub = rospy.Publisher("/predicted_trajectories", PredictedTrajectories, queue_size=1000)

  rate = rospy.Rate(7.5)
  while not rospy.is_shutdown():
    pub.publish(traj) 
    rate.sleep()
"""
    
#----------------------------------------------------------------------------------------------------------------------------------

def get_cmap(n, name='hsv'):
  return plt.cm.get_cmap(name, n)

def tracker_callback(msg, args):
    rospy.loginfo("Returning tracked people data")

    msg_list, model, pub, device, seq_length, scaler, points, dbscan, j = args
    
    #rospy.loginfo(msg)
    #rospy.loginfo("------------------------------------------------------------------------------------------------------------------")
    
    #msg = data

    for detection in msg.tracks:
      exists = False
      #rospy.loginfo(person)
      #rospy.loginfo("------------------------------------------------------------------------------------------------------------------")
      for person in msg_list:
        rospy.loginfo(len(person.path))

        #points.append([person.path[0,0], person.path[0,1]])

        #--------------------------------------------------------------------------------------------------------------------------
        """
        if len(points) > 1:
          labels = dbscan.fit_predict(points)
          for h in range(len(points)):
            if points[h][3] == person.id:
              person.cluster = labels[h]
          silhouette = silhouette_score(points, labels)
          davies_bouldin = davies_bouldin_score(points, labels)
          calinski_harabasz = calinski_harabasz_score(points, labels)
          dunn = dunn_index(points, labels)
        """

        #--------------------------------------------------------------------------------------------------------------------------

        if detection.track_id == person.id:
          rospy.loginfo("ID matched")
          #print(detection.color)
          exists = True
          person.add_detection(detection.pose.pose.position.x, detection.pose.pose.position.y, detection.pose.pose.orientation.w)#, detection.pose.pose.position.z)
          #print(detection.path)
          rospy.loginfo("Check")

          if len(person.path) == seq_length:
            rospy.loginfo("Time to predict!")
            start = time.time()

            path = np.array(person.path)
            #person.path.reshape(1, seq_length, 3)
            #rospy.loginfo(person.path)

            #coord = scaler.transform(path.reshape(-1,2)).reshape(path.shape)
            coord = torch.tensor(path, dtype=torch.float32)
            #rospy.loginfo(coord)

            coord = coord.view(1, seq_length, 2)
            coord = coord.to(device)
            rospy.loginfo(coord)
            rospy.loginfo("Data ready")
            
            output = model(coord)

            #data = list(zip(person.path, coord.cpu().detach().numpy(),output.cpu().detach().numpy(), itertools.repeat(msg.header.stamp, len(person.path))))
            #writer.writerows(data)
            #output = output.cpu().detach().numpy()
            #output = scaler.inverse_transform(output.reshape(-1,2)).reshape(output.shape)
            #del(coord)

            rospy.loginfo(output)
            rospy.loginfo("Output ready")
            prediction = PredictedTrajectory()
            color = ColorRGBA()
            color.r = person.id / 10.0  # Red component based on the ID
            color.g = 0.0
            color.b = 1.0 - (person.id / 10.0)  # Blue component based on the ID
            color.a = 1.0  # Alpha (transparency)
            prediction.track_id = person.id
            prediction.trajectory.header.frame_id = 'base_footprint'
            #prediction.trajectory = [Pose() for k in range(seq_length)]
            for i in range(seq_length):
              pose = Pose()
              #pose.color = color
              pose.position.x = output[0,i,0]
              pose.position.y = output[0,i,1]
              #prediction.trajectory[i].position.z = output[0,i,2]
              pose.position.z = 0

              pose.orientation.x = 0
              pose.orientation.y = 0
              pose.orientation.z = 0
              pose.orientation.w = person.w

              prediction.trajectory.poses.append(pose)
              
            rospy.loginfo("Prediction ready")
            pub.publish(prediction)
            pub2.publish(prediction.trajectory)
            rospy.loginfo("Prediction published")
            end = time.time()
            print(str(end - start))

            #person.path.pop(0)
            #person.counter = person.counter - 1

            #rospy.loginfo(person.path)
            
            person.path.pop(0)
            person.counter = person.counter - 1

            rospy.loginfo(person.counter)

      if not exists:
          rospy.loginfo("New ID")
          temp = Tracked(detection.track_id, seq_length)
          temp.add_detection(detection.pose.pose.position.x, detection.pose.pose.position.y, detection.pose.pose.orientation.w)#, detection.pose.pose.position.z)
          temp.counter = temp.counter + 1
          points.append([temp.path[0][0], temp.path[0][1], temp.id])
          msg_list.append(temp)

# Main

if __name__ == "__main__":

  rospy.init_node("trajectory_publisher")

  pub = rospy.Publisher("/predicted_trajectories", PredictedTrajectory, queue_size=1000)

  pub2 = rospy.Publisher("/array_of_poses", PoseArray, queue_size = 1000)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  csv_file_path = 'experiment.csv'

  #with open(csv_file_path, mode='w', newline='') as file:
    
    #writer = csv.writer(file)

    #writer.writerow(["Point", "Actual","Predicted", "Timestamp"])

  input_dim = 3
  num_layers = 2
  seq_length = 35
  hidden_size = 128
  msg_list = []
  points = []
  exists = False
  eps = 0.5
  min_samples = 2
  j = 0
    
  scaler = load(open('scaler_2d_final.pkl', 'rb'))
  dbscan = DBSCAN(eps=eps, min_samples=min_samples)

  model = LSTM_Trainer(2, 128, 3, 35)
  model.load_state_dict(torch.load("models/to_try/3lstm_3h128_2d_20_unscaled.pt"))
  model.to(device)

  model.eval()

  sub = rospy.Subscriber("/spencer/perception/tracked_persons_confirmed_by_upper_body", TrackedPersons, tracker_callback, (msg_list, model, pub, device, seq_length, scaler, points, dbscan, j))

  #rospy.spin()


  """
  #tracker = TrackerListener()
  while not rospy.is_shutdown():
    #print(tracker.get_data())
    #msg = tracker.get_data()

    for message in msg_list:
      for detection in message.tracks:
        print("Inside")
        if msg.tracks.track_id == detection.id:
          exists = True
          detection.add_detection(msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z)
          #print(detection.path)

          if detection.get_counter(seq_length):
            start = time.time()

            detection.path = np.array(detection.path)
            detection.path.reshape(1, seq_length, 3)

            detection.path = torch.tensor(detection.path, dtype=torch.float32)
            detection.path = detection.path.to(device)

            output = model(detection.path)

            prediction = PredictedTrajectory()

            prediction.track_id = detection.id
            for i in range(seq_length):
              prediction.trajectory[i].position.x = output[1,i,0]
              prediction.trajectory[i].position.y = output[1,i,1]
              prediction.trajectory[i].position.z = output[1,i,2]

              prediction.trajectory[i].pose.x = 0
              prediction.trajectory[i].pose.y = 0
              prediction.trajectory[i].pose.z = 0
              prediction.trajectory[i].pose.w = 0
            
            pub.publish(prediction)

            end = time.time()
            print(str(start - end))

            detection.path.pop(0)
            detection.counter = seq_length - 1

        else:
          print("Inside")
          temp = Tracked(msg.tracks.track_id, seq_length)
          temp.add_detection(msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z)
          temp.counter = temp.counter + 1
          msg_list.append(temp)



  #tracked_persons = tracker_subscriber()
"""
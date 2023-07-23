#!/usr/bin/env python3

import rospy
from lstm_trainer2 import LSTM_Trainer
import torch
import torch.nn
from spencer_tracking_msgs.msg import TrackedPersons
from spencer_tracking_msgs.msg import TrackedPerson
from trajectory_prediction.msg import PredictedTrajectories
from trajectory_prediction.msg import PredictedTrajectory

def tracker_callback(data):
    rospy.loginfo("Returning tracked people data")
    id = data.tracks[1].track_id
    rospy.loginfo(id)
    return data

def tracker_subscriber():
    
    rospy.init_node("tracker_subscriber")

    rospy.Subscriber("/spencer/perception/tracked_persons", TrackedPersons, tracker_callback)

    rospy.spin()

def trajectory_publisher(traj):
    
    rospy.init_node("trajectory_publisher")

    pub = rospy.Publisher("/predicted_trajectories", PredictedTrajectories, queue_size=1000)

    rate = rospy.Rate(7.5)
    while not rospy.is_shutdown():
        pub.publish(traj)
        rate.sleep()

if __name__ == "__main__":
    
    tracked_persons = tracker_subscriber()

    input_dim = 3
    num_layers = 3
    seq_length = 35
    hidden_size = 3

    model = LSTM_Trainer(input_dim, num_layers, seq_length, hidden_size)
    model.load_model("models/simple_model_0.pth")

    model.eval()

    with torch.no_grad():
        output = model(tracked_persons)

        trajectory_publisher(output)


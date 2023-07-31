#!/usr/bin/env python3

import rospy
from rospy import Time
import time
from rwth_perception_people_msgs.msg import GroundPlane

def ground_plane_publisher():
    rospy.init_node('ground_plane_publisher', anonymous=True)
    ground_plane_pub = rospy.Publisher('ground_plane', GroundPlane, queue_size=10)

  # Ground plane coefficients (A, B, C, D) and frame ID
    A = 0.0
    B = 0.0
    C = 1.0
    D = 0.0
    frame_id = "xtion_link"

    rate = rospy.Rate(10)  # 10 Hz

    while not rospy.is_shutdown():
        ground_plane_msg = GroundPlane()
        ground_plane_msg.header.seq = A
        t = rospy.Time.from_sec(time.time())
        ground_plane_msg.header.stamp = t
        #ground_plane_msg.header.stamp.nsec = t.to_nsec()
        ground_plane_msg.header.frame_id = frame_id
        ground_plane_msg.n = [-0.01958178409087674, -0.9997500170828265, -0.010791527913442955]
        ground_plane_msg.d = 1.59860336466
        
        ground_plane_pub.publish(ground_plane_msg)

        rate.sleep()

if __name__ == '__main__':
    try:
        ground_plane_publisher()
    except rospy.ROSInterruptException:
        pass


"""
header: 
  seq: 228
  stamp: 
    secs: 1439557603
    nsecs: 969129885
  frame_id: "rgbd_front_top_depth_optical_frame"
n: [-0.01958178409087674, -0.9997500170828265, -0.010791527913442955]
d: 1.59860336466
"""
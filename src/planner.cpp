#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <trajectory_prediction/msg/PredictedTrajectories.h>
#include <trajectory_prediction/msg/PredictedTrajectory.h>

void sub_callback(const std_msgs::StringConstPtr& str){
  int id = str.pred_trajs[0];
}

int main(int argc, char** argv)
{
  // Initialize the ROS node
  ros::init(argc, argv, "path_planner_subscriber");
  ros::NodeHandle nh;

  // Create a publisher to publish the planned path
  //ros::Publisher path_pub = nh.advertise<nav_msgs::Path>("planned_path", 10);

  // Create a subscriber to the predicted trajectories
  ros::Subscriber sub = nh.subscribe("predicted_trajectories",100,sub_callback)

  // Create a rate object to specify the publishing frequency (1 Hz in this example)
  ros::Rate rate(1);

  while (ros::ok())
  {
    // Create a Path message to represent the planned path
    nav_msgs::Path path_msg;

    // Populate the path with some sample waypoints
    for (int i = 0; i < 10; ++i)
    {
      geometry_msgs::PoseStamped pose;
      pose.pose.position.x = i;
      pose.pose.position.y = i * 0.5;
      pose.pose.orientation.w = 1.0;
      path_msg.poses.push_back(pose);
    }

    // Set the frame ID of the path (e.g., "map" or "odom")
    path_msg.header.frame_id = "map";

    // Publish the planned path
    path_pub.publish(path_msg);

    // Spin once to trigger the callbacks and update the ROS network
    ros::spinOnce();

    // Sleep to maintain the specified publishing rate
    rate.sleep();
  }

  return 0;
}

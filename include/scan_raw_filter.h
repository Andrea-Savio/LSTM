/*
 * Header file for the scan_raw_filter.
 */

#ifndef SCAN_RAW_FILTER_H
#define SCAN_RAW_FILTER_H

#include <ros/ros.h> //ROS Default Header File
#include <string> //Header file to use string type
#include <std_msgs/String.h> //Header file to use string type
#include <sensor_msgs/LaserScan.h> //Header file to use LaserScan type
#include <vector> //Header file to use the vector type

template<typename PublisherType, typename SubscriberType>
class Scan_Raw_Filter
{
public:
	static const int LASER_POINTS_TO_SKIP = 20; //Laser points to skip both at the beginning of each measure and at the end
    const float NULL_LASER_INTENSITY = 0.00; //Null intensity to set by default
	Scan_Raw_Filter(){}
	Scan_Raw_Filter(std::string publishTopicName, std::string subscribeTopicName, int queueSize) 
	{
		filter_publisher = n.advertise<PublisherType>(publishTopicName, queueSize);
		filter_publisher_base = n.advertise<PublisherType>("base_scan", queueSize);
		filter_subscriber = n.subscribe<SubscriberType>(subscribeTopicName, queueSize, &Scan_Raw_Filter::subscriberCallback,this);
	}
	void subscriberCallback(const typename SubscriberType::ConstPtr& receivedMsg);

protected:
	ros::Subscriber filter_subscriber;
	ros::Publisher filter_publisher;
	ros::Publisher filter_publisher_base;
	ros::NodeHandle n;
};

#endif

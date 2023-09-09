#include "ros/ros.h" //ROS Default Header File
#include "scan_raw_filter.h" //Header file for this type of node

template<>
void Scan_Raw_Filter<sensor_msgs::LaserScan, sensor_msgs::LaserScan>::subscriberCallback(const sensor_msgs::LaserScan::ConstPtr& old_scan)
{
	//To filter the readed message
	sensor_msgs::LaserScan filtered_scan;

	//Copy the info of the message in a new message,
	//keeping the same info about header, frame_id, angle_increment,
	//time_increment, scan_time,range_min, range_max;
	//but filter the first and the last points and compute the new
	//angle_min and angle_max
	filtered_scan.header.seq = old_scan->header.seq;
	filtered_scan.header.stamp.sec = old_scan->header.stamp.sec;
	filtered_scan.header.stamp.nsec = old_scan->header.stamp.nsec;
	filtered_scan.header.frame_id = old_scan->header.frame_id;
	filtered_scan.angle_min = old_scan->angle_min + LASER_POINTS_TO_SKIP * old_scan->angle_increment;
	filtered_scan.angle_max = old_scan->angle_max - LASER_POINTS_TO_SKIP * old_scan->angle_increment;
	filtered_scan.angle_increment = old_scan->angle_increment;
	filtered_scan.time_increment = old_scan->time_increment;
	filtered_scan.scan_time = old_scan->scan_time;
	filtered_scan.range_min = old_scan->range_min;
	filtered_scan.range_max = old_scan->range_max;

	//Here the real filter on range data is applied
	for(int i=LASER_POINTS_TO_SKIP; i < (old_scan->ranges.size() - LASER_POINTS_TO_SKIP); i++)
	{
		filtered_scan.ranges.push_back(old_scan->ranges[i]);
	}

	//Here the real filter on intensities data is applied
	/*for(int i=LASER_POINTS_TO_SKIP; i < (old_scan->intensities.size() - LASER_POINTS_TO_SKIP); i++)
	{//ROS_INFO("try to add new intensity %d", i);
		//filtered_scan.intensities.push_back(old_scan->intensities[i]);ROS_INFO("added new intensity %d", i);
	}*/
	for(int i=LASER_POINTS_TO_SKIP; i < (old_scan->ranges.size() - LASER_POINTS_TO_SKIP); i++)
	{
		filtered_scan.intensities.push_back(Scan_Raw_Filter::NULL_LASER_INTENSITY);
	}
	
	//To publish the filtered message
	filter_publisher.publish(filtered_scan);//ROS_INFO("end modify message");
}

int main(int argc, char **argv) //Main function
{
	ros::init(argc, argv, "scan_raw_filter"); //Node initialization
	//Filter object
	Scan_Raw_Filter<sensor_msgs::LaserScan, sensor_msgs::LaserScan> scan_raw_filter("scan_raw_filtered","scan_raw",1000);
	ros::spin(); //ROS spin

	return 0; //Exit
}

# Message for predicted trajectories of a tracked person
#

uint64                          track_id 		# unique identifier of the target, consistent over time

geometry_msgs/Pose[]            trajectory 		# predicted trajectory for n seconds (7.5 Hz)

string context						# predicted context

# Import necessary ROS packages and message types
import rospy
from geometry_msgs.msg import Pose, PoseArray
from spencer_tracking_msgs.msg import TrackedPersons, DetectedPersons
from spencer_tracking_msgs.msg import TrackedPerson, DetectedPerson
from std_msgs.msg import Float64  # For the confidence score

# Global dictionary to store detected_person messages with confidence scores
detected_persons = {}
tracked_persons = TrackedPersons()

def tracked_person_callback(msg):
    # Check if the detection_id exists in the dictionary
    for i in range(len(msg)):
        detection_id = msg[i].detection_id
        if detection_id in detected_persons:
            # Get the confidence score associated with the detection_id
            confidence_score = detected_persons[detection_id]
            # Check if the confidence score is above 0.5
            if confidence_score > 0.5:
                tracked_persons.tracks.append(msg[i])
                # Publish the tracked_person_msg
            
    tracked_person_publisher.publish(tracked_persons)
    tracked_persons = TrackedPersons()

def detected_person_callback(msg):
    # Store the confidence score in the dictionary with the detection_id as the key
    detection_id = msg.detection_id
    confidence_score = msg.confidence
    detected_persons[detection_id] = confidence_score

if __name__ == '__main__':
    rospy.init_node('/filter_tracked_persons')
    
    # Create a publisher for the filtered tracked_person messages
    tracked_person_publisher = rospy.Publisher('/filtered_tracked_persons', TrackedPersons, queue_size=1000)

    # Subscribe to the tracked_person and detected_person topics
    sub_track = rospy.Subscriber('/spencer/perception/tracked_persons_confirmed_by_upper_body', TrackedPersons, tracked_person_callback)
    sub_det = rospy.Subscriber('/spencer/perception/detected_persons', DetectedPersons, detected_person_callback)

    rospy.spin()  # Keep the node running

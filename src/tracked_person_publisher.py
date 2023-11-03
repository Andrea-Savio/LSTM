import rospy
import time
from geometry_msgs.msg import PoseWithCovariance, TwistWithCovariance, AccelWithCovariance
from spencer_tracking_msgs.msg import TrackedPersons
from spencer_tracking_msgs.msg import TrackedPerson
from human_msgs.msg import TrackedHumans, TrackedHuman, TrackedSegment
from std_msgs.msg import Header

def tracked_person_callback(msg,args):

    pub = args
    cohan = TrackedHumans()

    cohan.header.seq = msg.header.seq
    cohan.header.stamp = msg.header.stamp
    cohan.header.frame_id = msg.header.frame_id

    cohan.humans = []
    ids = []

    for i in range(len(msg.tracks)):
        human = TrackedHuman()
        human.segments = [TrackedSegment()]
        seg = TrackedSegment()
        if (msg.tracks[i].track_id not in ids):
            human.track_id = msg.tracks[i].track_id
            ids.append(msg.tracks[i].track_id)
            human.state = 1
            seg.type = 1
            seg.pose = msg.tracks[i].pose
            seg.twist = msg.tracks[i].twist
            #seg.accel =
            human.segments.append(seg)

            cohan.humans.append(human)

    pub.publish(cohan)
    print("Published humans!")



if __name__ == '__main__':
    rospy.init_node('tracked_person_publisher')

    pub = rospy.Publisher("/tracked_humans", TrackedHumans, queue_size=1000)
    sub = rospy.Subscriber("/spencer/perception/tracked_persons", TrackedPersons, tracked_person_callback, (pub))

    rospy.spin()

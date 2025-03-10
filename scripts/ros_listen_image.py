import os
import sys
import copy
import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
from scipy.spatial.transform import Rotation as Rot
import time
import json
dynamic_path = os.path.abspath(__file__+"/../")
# print(dynamic_path)
sys.path.append(dynamic_path)

img_folder = os.path.join(dynamic_path, 'image')

if not os.path.exists(img_folder):
    os.makedirs(img_folder)

bridge = CvBridge()

class dVRKImageSubscriber:
    def __init__(self):
        rospy.init_node('goovis_img_sub', anonymous=True)
        self.camera1_topic = '/dvrk_si/left/image_raw'
        self.camera2_topic = '/dvrk_si/right/image_raw'
        self.camera1_img = None
        self.camera2_img = None
        self.record_status = False
        self.sub_topic_camera1 = rospy.Subscriber(self.camera1_topic, Image, self.camera1_sub, queue_size=1)
        self.sub_topic_camera2 = rospy.Subscriber(self.camera2_topic, Image, self.camera2_sub, queue_size=1)
        self.count = 0

    def camera1_sub(self, msg):
        # rospy.loginfo("camera left read")
        try:
            self.camera1_img = bridge.imgmsg_to_cv2(msg, "bgr8")
            # print(self.camera1_img)
            # print('camera left read')
        except CvBridgeError as e:
            print(e)

    def camera2_sub(self, msg):
        # rospy.loginfo("camera right read")
        try:
            self.camera2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
            # print("camera right read")
        except CvBridgeError as e:
            print(e)

    def save_img(self, folder_path, img_left, img_right):
        img_left = copy.deepcopy(img_left)
        img_right = copy.deepcopy(img_right)
        img_left_name = f'{self.count}_left.jpg'
        img_right_name = f'{self.count}_right.jpg'
        img_left_path = os.path.join(folder_path, img_left_name)
        img_right_path = os.path.join(folder_path, img_right_name)
        cv2.imwrite(img_left_path, img_left)
        cv2.imwrite(img_right_path, img_right)

    def get_data(self):
        return self.camera1_img, self.camera2_img

    def sub_run(self, feq):
        rate = rospy.Rate(feq)
        time.sleep(1)
        while not rospy.is_shutdown():
            img_left, img_right = self.get_data()
            print('saving')
            self.save_img(img_folder, img_left, img_right)
            self.count += 1
            input("Press Enter to Continue")
            if self.count >= 30:
                break
            rate.sleep()

if __name__=="__main__":
    refresh_rate = 1
    sub_cls = dVRKImageSubscriber()
    sub_cls.sub_run(refresh_rate)
    print('Done')
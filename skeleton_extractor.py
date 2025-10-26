import cv2
import numpy as np
import sys
import os
import argparse

from joint_angles import convert_to_joint_angles
from Max_skeleton import select_max_bounding_area_skeleton
from itertools import zip_longest


MODEL_FOLDER = r"C:\Users\ATRI SUKUL\Desktop\IIT Tirupati\ISL\Project\openpose\models"

# OpenPose paths (required for importing)
sys.path.append(r"C:\Users\ATRI SUKUL\Desktop\IIT Tirupati\ISL\Project\openpose\build\python\openpose\Release")
sys.path.append(r"C:\Program Files\OpenPose\python")
os.environ['PATH'] += r';C:\Users\ATRI SUKUL\Desktop\IIT Tirupati\ISL\Project\openpose\build\bin;C:\Users\ATRI SUKUL\Desktop\IIT Tirupati\ISL\Project\openpose\build\x64\Release'

import pyopenpose as op


class PoseEstimator:

    def __init__(self, openpose_path):

        
        self.openpose_path = openpose_path
        self.params = {}
        self.opWrapper = None
        

        self._init_openpose()
    
    def _init_openpose(self):

        try:

            
            # parameters
            self.params["model_folder"] = MODEL_FOLDER
            self.params["model_pose"] = "BODY_25"
            self.params["number_people_max"] = -1  # detect all people in the image
            
            # Initialize OpenPose wrapper
            self.opWrapper = op.WrapperPython()
            self.opWrapper.configure(self.params)
            self.opWrapper.start()
            
            print("OpenPose initialized with body_25 model")
            
        except Exception as e:
            print(f"Error initializing OpenPose: {e}")
    
    def extract_keys_25(self, image):

        if self.opWrapper is None:
            raise RuntimeError("OpenPose not initialized")
        
        # Create datum object
        datum = op.Datum()
        datum.cvInputData = image
        
        # Process image
        self.opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        
        # Extract keypoints
        pose_keypoints = datum.poseKeypoints
        
        if pose_keypoints is None or len(pose_keypoints) == 0:
            print("No people detected in the image")
            return []
        

        skeletons = []
        for i in range(pose_keypoints.shape[0]):
            skeletons.append(pose_keypoints[i])
        
        print(f"Detected {len(skeletons)} person(s) in the image")
        return skeletons


    def visualize_skeletons(self,image, skeleton, skeletons, detect_all=False, output_path="output.jpg"):

        # BODY_25 skeleton connections
        def grouper(iterable, n, fillvalue=None):
            
            
            args = [iter(iterable)] * n
            return list(zip_longest(*args, fillvalue=fillvalue))
        
        # fetches a flat list of keypoint indices defining the skeleton connections (e.g., [1,0, 1,2, 1,5, ...]) for the BODY_25 model, from the OpenPose library.
        POSE_PAIRS = op.getPosePartPairs(op.BODY_25)
        # splits that flat list into consecutive pairs/tuples (e.g., [(1,0), (1,2), (1,5), ...])
        POSE_PAIRS = grouper(POSE_PAIRS, 2, None)
        
        output_image = image.copy()
        
        # Draw keypoints
        for i in range(skeleton.shape[0]):
            if skeleton[i, 2] > 0.1:  # confidence threshold
                x, y = int(skeleton[i, 0]), int(skeleton[i, 1])
                cv2.circle(output_image, (x, y), 5, (0, 255, 255), -1)
        
        # if we want to detect all the skeletons in the image
        if detect_all:
            for _, skeleton in enumerate(skeletons):
                # Drawing skeleton
                for pair in POSE_PAIRS:
                    a, b = pair
                    if skeleton[a, 2] > 0.1 and skeleton[b, 2] > 0.1:
                        x1, y1 = int(skeleton[a, 0]), int(skeleton[a, 1])
                        x2, y2 = int(skeleton[b, 0]), int(skeleton[b, 1])
                        cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            output_path = output_path.split('.')[0] + '_all.jpg'
        else:
            for pair in POSE_PAIRS:
                a, b = pair
                if skeleton[a, 2] > 0.1 and skeleton[b, 2] > 0.1:
                    x1, y1 = int(skeleton[a, 0]), int(skeleton[a, 1])
                    x2, y2 = int(skeleton[b, 0]), int(skeleton[b, 1])
                    cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imwrite(output_path, output_image)
        print(f"Pose visualization saved to: {output_path}")
        
        return output_image

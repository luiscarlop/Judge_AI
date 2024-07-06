from FSM_squat import FSM_squat
from thresholds import *

import cv2
import av
import numpy as np
import mediapipe as mp

from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

FSM = FSM_squat(get_thresholds())

class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.pose = mp.solutions.pose.Pose(static_image_mode=False)

        def transform(self, frame):
            frame = frame.to_ndarray(format="bgr24")

            # Process the frame
            new_frame = FSM_squat.process(frame, self.pose)

            return av.VideoFrame.from_ndarray(new_frame, format="bgr24")
            
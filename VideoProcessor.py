from FSM_squat import FSM_squat
from thresholds import *
from utils import draw_text, get_landmark_features, get_mediapipe_pose

import cv2
import av
import numpy as np
import mediapipe as mp

from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

FSM = FSM_squat(get_thresholds())

mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose

class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(static_image_mode=False)

        def transform(self, frame):
            frame = frame.to_ndarray(format="bgr24")

            # Process the frame
            FSM_squat.process(FSM, frame, self.pose)

            return frame
        

class PhotoProcessor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True)

        # Dictionary to maintain the various landmark features.
        self.dict_features = {}
        self.left_features = {
                                "shoulder": 11,
                                "elbow": 13,
                                "wrist": 15,
                                "hip": 23,
                                "knee": 25,
                                "ankle": 27,
                                "foot": 31,
        }

        self.right_features = {
                                "shoulder": 12,
                                "elbow": 14,
                                "wrist": 16,
                                "hip": 24,
                                "knee": 26,
                                "ankle": 28,
                                "foot": 32,
        }

        self.dict_features["left"] = self.left_features
        self.dict_features["right"] = self.right_features

    def process(self, np_image: np.ndarray):
        with self.pose as pose:
            try:
                image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
            except:
                image = np_image

            height, width, _ = image.shape
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = pose.process(image_rgb)

            if results.pose_landmarks is not None:

                ps_lm = results.pose_landmarks

                right_shoulder, right_elbow, right_wrist, right_hip, right_knee, right_ankle, right_foot = \
                    get_landmark_features(ps_lm.landmark, self.dict_features, "right", width, height)

                left_shoulder, left_elbow, left_wrist, left_hip, left_knee, left_ankle, left_foot = \
                    get_landmark_features(ps_lm.landmark, self.dict_features, "left", width, height)

                puntuacion = list()

                cv2.line(
                    image, left_shoulder, right_shoulder, (255, 255, 255), 3
                )  # unión de punto xy1 con xy2
                cv2.line(image, left_shoulder, left_elbow, (255, 255, 255), 3)
                cv2.line(
                    image, left_shoulder, left_hip, (255, 255, 255), 3
                )  # unión de punto xy2 con xy3
                cv2.line(image, left_elbow, left_wrist, (255, 255, 255), 3)
                cv2.line(image, right_shoulder, right_elbow, (255, 255, 255), 3)
                cv2.line(image, right_elbow, right_wrist, (255, 255, 255), 3)
                cv2.line(image, right_shoulder, right_hip, (255, 255, 255), 3)
                cv2.line(image, right_hip, right_knee, (255, 255, 255), 3)
                cv2.line(image, right_hip, left_hip, (255, 255, 255), 3)
                cv2.line(image, right_knee, right_ankle, (255, 255, 255), 3)
                cv2.line(image, left_hip, left_knee, (255, 255, 255), 3)
                cv2.line(image, left_knee, left_ankle, (255, 255, 255), 3)

                cv2.circle(image, left_shoulder, 6, (43, 214, 38), -1)  # color verde
                cv2.circle(image, right_shoulder, 6, (43, 214, 38), -1)  # color verde
                cv2.circle(image, left_elbow, 6, (43, 214, 38), -1)  # color verde
                cv2.circle(image, left_wrist, 6, (43, 214, 38), -1)  # color verde
                cv2.circle(image, right_elbow, 6, (43, 214, 38), -1)  # color verde
                cv2.circle(image, right_wrist, 6, (43, 214, 38), -1)  # color verde
                cv2.circle(image, right_hip, 6, (43, 214, 38), -1)  # color verde
                cv2.circle(image, right_knee, 6, (43, 214, 38), -1)  # color verde
                cv2.circle(image, right_ankle, 6, (43, 214, 38), -1)  # color verde
                cv2.circle(
                    image, left_hip, 6, (43, 214, 38), -1
                )  # color azul corrdenadas según indice del dibujo
                cv2.circle(
                    image, left_knee, 6, (43, 214, 38), -1
                )  # color azul corrdenadas según indice del dibujo
                cv2.circle(image, left_ankle, 6, (43, 214, 38), -1)

                if (
                    (right_hip[1] >= right_knee[1] - 30
                    and right_hip[1] <= right_knee[1] + 30)
                    or (left_hip[1] >= left_knee[1] - 30
                    and left_hip[1] <= left_knee[1] + 30)
                ):
                    cv2.line(
                        image, right_hip, right_knee, (0, 255, 255), 3
                    )  # unión de punto xy1 con xy2
                    cv2.line(
                        image, left_hip, left_knee, (0, 255, 255), 3
                    )  # unión de punto xy1 con xy2

                    puntos = 2
                    puntuacion.append(puntos)
                    print(f" puntuación: {sum(puntuacion)} ")

                    if sum(puntuacion) >= 2:

                        font = cv2.FONT_HERSHEY_SIMPLEX

                        draw_text(
                            image,
                            "PARALLEL POSITION",
                            pos=(int(width * 0.68), 30),
                            text_color=(255, 255, 230),
                            font_scale=0.7,
                            text_color_bg=(127, 233, 100)
                        )
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                return image_rgb
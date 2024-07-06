from FSM_squat import FSM_squat
from thresholds import *
from utils import draw_text

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

    def process(self, np_image: np.ndarray):
        with self.pose as pose:
            image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

            height, width, _ = image.shape
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = pose.process(image_rgb)

            if results.pose_landmarks is not None:

                x11 = int(results.pose_landmarks.landmark[11].x * width)
                y11 = int(results.pose_landmarks.landmark[11].y * height)

                x12 = int(results.pose_landmarks.landmark[12].x * width)
                y12 = int(results.pose_landmarks.landmark[12].y * height)

                x13 = int(results.pose_landmarks.landmark[13].x * width)
                y13 = int(results.pose_landmarks.landmark[13].y * height)

                x14 = int(results.pose_landmarks.landmark[14].x * width)
                y14 = int(results.pose_landmarks.landmark[14].y * height)

                x15 = int(results.pose_landmarks.landmark[15].x * width)
                y15 = int(results.pose_landmarks.landmark[15].y * height)

                x16 = int(results.pose_landmarks.landmark[16].x * width)
                y16 = int(results.pose_landmarks.landmark[16].y * height)

                # x17= int(results.pose_landmarks.landmark[17].x * width)
                # y17= int(results.pose_landmarks.landmark[17].y * height)

                # x18= int(results.pose_landmarks.landmark[18].x * width)
                # y18= int(results.pose_landmarks.landmark[18].y * height)

                # x19= int(results.pose_landmarks.landmark[19].x * width)
                # y19= int(results.pose_landmarks.landmark[19].y * height)

                # x20= int(results.pose_landmarks.landmark[20].x * width)
                # y20= int(results.pose_landmarks.landmark[20].y * height)

                # x21= int(results.pose_landmarks.landmark[21].x * width)
                # y21= int(results.pose_landmarks.landmark[21].y * height)

                # x22= int(results.pose_landmarks.landmark[22].x * width)
                # y22= int(results.pose_landmarks.landmark[22].y * height)

                x23 = int(results.pose_landmarks.landmark[23].x * width)
                y23 = int(results.pose_landmarks.landmark[23].y * height)

                x24 = int(results.pose_landmarks.landmark[24].x * width)
                y24 = int(results.pose_landmarks.landmark[24].y * height)

                x25 = int(results.pose_landmarks.landmark[25].x * width)
                y25 = int(results.pose_landmarks.landmark[25].y * height)

                x26 = int(results.pose_landmarks.landmark[26].x * width)
                y26 = int(results.pose_landmarks.landmark[26].y * height)

                x27 = int(results.pose_landmarks.landmark[27].x * width)
                y27 = int(results.pose_landmarks.landmark[27].y * height)

                x28 = int(results.pose_landmarks.landmark[28].x * width)
                y28 = int(results.pose_landmarks.landmark[28].y * height)

                puntuacion = list()

                cv2.line(
                    image, (x11, y11), (x12, y12), (255, 255, 255), 3
                )  # unión de punto xy1 con xy2
                cv2.line(image, (x11, y11), (x13, y13), (255, 255, 255), 3)
                cv2.line(
                    image, (x11, y11), (x23, y23), (255, 255, 255), 3
                )  # unión de punto xy2 con xy3
                cv2.line(image, (x13, y13), (x15, y15), (255, 255, 255), 3)
                cv2.line(image, (x12, y12), (x14, y14), (255, 255, 255), 3)
                cv2.line(image, (x14, y14), (x16, y16), (255, 255, 255), 3)
                cv2.line(image, (x12, y12), (x24, y24), (255, 255, 255), 3)
                cv2.line(image, (x24, y24), (x26, y26), (255, 255, 255), 3)
                cv2.line(image, (x24, y24), (x23, y23), (255, 255, 255), 3)
                cv2.line(image, (x26, y26), (x28, y28), (255, 255, 255), 3)
                cv2.line(image, (x23, y23), (x25, y25), (255, 255, 255), 3)
                cv2.line(image, (x25, y25), (x27, y27), (255, 255, 255), 3)

                cv2.circle(image, (x11, y11), 6, (43, 214, 38), -1)  # color verde
                cv2.circle(image, (x12, y12), 6, (43, 214, 38), -1)  # color verde
                cv2.circle(image, (x13, y13), 6, (43, 214, 38), -1)  # color verde
                cv2.circle(image, (x15, y15), 6, (43, 214, 38), -1)  # color verde
                cv2.circle(image, (x14, y14), 6, (43, 214, 38), -1)  # color verde
                cv2.circle(image, (x16, y16), 6, (43, 214, 38), -1)  # color verde
                cv2.circle(image, (x24, y24), 6, (43, 214, 38), -1)  # color verde
                cv2.circle(image, (x26, y26), 6, (43, 214, 38), -1)  # color verde
                cv2.circle(image, (x28, y28), 6, (43, 214, 38), -1)  # color verde
                cv2.circle(
                    image, (x23, y23), 6, (43, 214, 38), -1
                )  # color azul corrdenadas según indice del dibujo
                cv2.circle(
                    image, (x25, y25), 6, (43, 214, 38), -1
                )  # color azul corrdenadas según indice del dibujo
                cv2.circle(image, (x27, y27), 6, (43, 214, 38), -1)

                if (
                    (y24 >= y26 - 30
                    and y24 <= y26 + 30)
                    or (y23 >= y25 - 30
                    and y23 <= y25 + 30)
                ):
                    cv2.line(
                        image, (x24, y24), (x26, y26), (0, 255, 255), 3
                    )  # unión de punto xy1 con xy2
                    cv2.line(
                        image, (x23, y23), (x25, y25), (0, 255, 255), 3
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
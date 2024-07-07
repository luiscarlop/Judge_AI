import streamlit as st
import base64
from PIL import Image
import streamlit as st
import cv2
import numpy as np
import av
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import io
from judge_pose import Manual_Pose_with_array
from PIL import ImageFile
import threading


st.set_page_config(
    page_title="Judge AI",
    page_icon=":weight_lifter:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://www.judgeai.com/help",  # ejemplo de como sería una vez montado
        "Report a bug": "https://www.judgeai.com/bug",  # ejemplo de como sería una vez montado
        "About": "# This is JudgeAi. This is our Data Science Final Proyect !",
    },
)


st.markdown(
    "<h1 style='text-align: center; '>Welcome to the fun part</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<hr style='border:2px solid #f3b323; margin-bottom: 0px; margin-top: 0px'>",
    unsafe_allow_html=True,
)


st.markdown(
    "<h3 style='text-align: center; '>You can test our AI with 3 freatures</h3>",
    unsafe_allow_html=True,
)

# st.write(
#     "<p style='text-align: center;'>- 1. Live video</p>", unsafe_allow_html=True
# )
# st.write("<p style='text-align: center;'>- 2. Selfie mode</p>", unsafe_allow_html=True)
# st.write(
#     "<p style='text-align: center;'>- 3. Upload a photo</p>", unsafe_allow_html=True
# )
# st.title("Welcome to the fun part")
# st.subheader("You can test our AI with 3 freatures")

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.write(
        """
- 1. Real-time video
"""
    )

with col2:
    st.write(
        """
- 2. Selfie mode
"""
    )


with col3:
    st.write(
        """
- 3. Upload a photo
"""
    )

col1, col2 = st.columns([1, 1.5])

# st.write(
#     """
# - 1. Real-time video
# - 2. Selfie mode
# - 3. Upload a photo
# """
# )

st.write("This page is dedicated to our own trained AI")

st.write("More modes, like competition mode and youtube video are coming soon!")


st.write(
    "This AI is very sensitve, so if possible reduce ammount of people animals or objects on screen."
)
st.write("Navigate trough the modes to try them:")


def main_page():
    # st.markdown("# 🏠Home")
    st.sidebar.markdown("# 🏠Home")


def page2():
    # st.markdown("#🕺Our model")
    st.sidebar.markdown("# 🕺Our model")


def page3():
    # st.markdown("# ⚖️Pre-trained model")
    st.sidebar.markdown("# ⚖️Pre-trained model")


def page4():
    # st.markdown("# 🕵️‍♂️About_us")
    st.sidebar.markdown("# 🕵️‍♂️About_us")


page_names_to_funcs = {
    "🏠Home": main_page,
    "🕺Our model": page2,
    "⚖️Pre-trained model": page3,
    "🕵️‍♂️About_us": page4,
}

# selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
# page_names_to_funcs[selected_page]()


option = st.selectbox(
    "Select an option:",
    ("Intoduction", "Live video", "Photo mode", "Upload a photo"),
    key=1,
)
if option == "Intoduction":
    st.subheader("Before starting")
    st.write(
        "If you are going to use the selfie mode or to film youself follow this steps first:"
    )
    st.write(
        """
    - 1. Set your workout space properly (prepare your camera, your weights, etc.).
    - 2. Once you are ready, select the option on the app.
    - 3. Enjoy the feature selected!
    """
    )
    st.write("If you just want to upload a photo, you can go straight to it.")
    st.write("Let's go!")

ImageFile.LOAD_TRUNCATED_IMAGES = True
if option == "Live video":
    #######################################################################################
    # st.subheader("Live video")
    # st.write("Click on the start button and start doing squats.")
    # st.write("When you squat low enough, an indicator will appear on the screen")
    # st.write(
    #     "Throughout the process, you will receive outputs indicating the state of your squat. \
    # Once you finish your squat, it will be added to your score."
    # )
    # mp_drawing = mp.solutions.drawing_utils
    # manual_pose = Manual_Pose_with_array()

    # class VideoProcessor(VideoTransformerBase):
    #     def __init__(self):
    #         self.pose = Manual_Pose_with_array().pose_landmarks
    #         self.lock = threading.Lock()
    #         self.img_container = {"img": None}

    #     def video_frame_callback(self, frame):
    #         frame = frame.to_ndarray(format="bgr24")
    #         height, width, _ = frame.shape
    #         results = self.pose.process(frame)
    #         if results.pose_landmarks is not None:
    #             x0 = int(self.pose["x"][0])
    #             y0 = int(self.pose["y"][0])
    #             x1 = int(self.pose["x"][1])
    #             y1 = int(self.pose["y"][1])
    #             x2 = int(self.pose["x"][2])
    #             y2 = int(self.pose["y"][2])
    #             x3 = int(self.pose["x"][3])
    #             y3 = int(self.pose["y"][3])
    #             x4 = int(self.pose["x"][4])
    #             y4 = int(self.pose["y"][4])
    #             x5 = int(self.pose["x"][5])
    #             y5 = int(self.pose["y"][5])
    #             x6 = int(self.pose["x"][6])
    #             y6 = int(self.pose["y"][6])
    #             x7 = int(self.pose["x"][7])
    #             y7 = int(self.pose["y"][7])
    #             x8 = int(self.pose["x"][8])
    #             y8 = int(self.pose["y"][8])
    #             x9 = int(self.pose["x"][9])
    #             y9 = int(self.pose["y"][9])
    #             cv2.line(frame, (x0, y0), (x3, y3), (255, 255, 255), 3)
    #             cv2.line(frame, (x1, y1), (x3, y3), (255, 255, 255), 3)
    #             cv2.line(frame, (x2, y2), (x3, y3), (255, 255, 255), 3)
    #             cv2.line(frame, (x4, y4), (x5, y5), (255, 255, 255), 3)
    #             cv2.line(frame, (x4, y4), (x0, y0), (255, 255, 255), 3)
    #             cv2.line(frame, (x5, y5), (x1, y1), (255, 255, 255), 3)
    #             cv2.line(frame, (x4, y4), (x6, y6), (255, 255, 255), 3)
    #             cv2.line(frame, (x5, y5), (x7, y7), (255, 255, 255), 3)
    #             cv2.line(frame, (x6, y6), (x8, y8), (255, 255, 255), 3)
    #             cv2.line(frame, (x7, y7), (x9, y9), (255, 255, 255), 3)
    #             puntuacion = []
    #             if y4 >= y6 - 30 and y4 <= y6 + 30 or y5 >= y7 - 30 and y5 <= y7 + 30:
    #                 cv2.line(frame, (x4, y4), (x6, y6), (0, 255, 255), 3)
    #                 cv2.line(frame, (x5, y5), (x7, y7), (0, 255, 255), 3)
    #                 puntos = 2
    #                 puntuacion.append(puntos)
    #             if sum(puntuacion) >= 2:
    #                 font = cv2.FONT_HERSHEY_SIMPLEX
    #                 cv2.putText(
    #                     img=frame,
    #                     text="PARALLEL POSITION",
    #                     org=(int(width * 0.68), 30),
    #                     fontFace=font,
    #                     fontScale=0.7,
    #                     color=(255, 255, 230),
    #                     thickness=2,
    #                     lineType=cv2.LINE_AA,
    #                 )
    #             with self.lock:
    #                 self.img_container["img"] = frame
    #         return frame

    # webrtc_streamer(key="example", video_transformer_factory=VideoProcessor)
    soon = cv2.imread("data/stored_pictures/soon.jpg")
    soon_rgb = cv2.cvtColor(soon, cv2.COLOR_BGR2RGB)
    st.image(soon_rgb)
else:
    st.write("No image captured. Please capture an image to proceed.")


###################################################################################################

if option == "Photo mode":

    st.subheader("Photo mode")
    st.write(
        "Take a photo when you’re at the lowest point of your squat. Please ask someone at the gym for help to avoid injuries!"
    )
    st.write("Once the photo is taked you can check how good you were doing it.")
    st.write(
        "Even if this process isn’t in real-time, you’ll earn a point if your photo shows that you’ve gone bellow the parallel position in your squat."
    )
    img_file_buffer = st.camera_input("Take a picture!")
    manual_pose = Manual_Pose_with_array()
    if img_file_buffer is not None:
        # Convert the image to a numpy array
        np_image = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)

        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        # image = cv2.resize(src=image, dsize=(642, 1141))
        height, width, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        manual_pose.process(image_rgb)
        if manual_pose.pose_landmarks is not None:
            x0 = int(manual_pose.pose_landmarks["x"][0])
            y0 = int(manual_pose.pose_landmarks["y"][0])
            x1 = int(manual_pose.pose_landmarks["x"][1])
            y1 = int(manual_pose.pose_landmarks["y"][1])
            x2 = int(manual_pose.pose_landmarks["x"][2])
            y2 = int(manual_pose.pose_landmarks["y"][2])
            x3 = int(manual_pose.pose_landmarks["x"][3])
            y3 = int(manual_pose.pose_landmarks["y"][3])
            x4 = int(manual_pose.pose_landmarks["x"][4])
            y4 = int(manual_pose.pose_landmarks["y"][4])
            x5 = int(manual_pose.pose_landmarks["x"][5])
            y5 = int(manual_pose.pose_landmarks["y"][5])
            x6 = int(manual_pose.pose_landmarks["x"][6])
            y6 = int(manual_pose.pose_landmarks["y"][6])
            x7 = int(manual_pose.pose_landmarks["x"][7])
            y7 = int(manual_pose.pose_landmarks["y"][7])
            x8 = int(manual_pose.pose_landmarks["x"][8])
            y8 = int(manual_pose.pose_landmarks["y"][8])
            x9 = int(manual_pose.pose_landmarks["x"][9])
            y9 = int(manual_pose.pose_landmarks["y"][9])
            cv2.line(manual_pose.image, (x0, y0), (x3, y3), (255, 255, 255), 3)
            cv2.line(manual_pose.image, (x1, y1), (x3, y3), (255, 255, 255), 3)
            cv2.line(manual_pose.image, (x2, y2), (x3, y3), (255, 255, 255), 3)
            cv2.line(manual_pose.image, (x4, y4), (x5, y5), (255, 255, 255), 3)
            cv2.line(manual_pose.image, (x4, y4), (x0, y0), (255, 255, 255), 3)
            cv2.line(manual_pose.image, (x5, y5), (x1, y1), (255, 255, 255), 3)
            cv2.line(manual_pose.image, (x4, y4), (x6, y6), (255, 255, 255), 3)
            cv2.line(manual_pose.image, (x5, y5), (x7, y7), (255, 255, 255), 3)
            cv2.line(manual_pose.image, (x6, y6), (x8, y8), (255, 255, 255), 3)
            cv2.line(manual_pose.image, (x7, y7), (x9, y9), (255, 255, 255), 3)
            puntuacion = []
            if y4 >= y6 - 30 and y4 <= y6 + 30 or y5 >= y7 - 30 and y5 <= y7 + 30:
                cv2.line(
                    manual_pose.image, (x4, y4), (x6, y6), (0, 255, 255), 3
                )  # unión de punto xy1 con xy2
                cv2.line(
                    manual_pose.image, (x5, y5), (x7, y7), (0, 255, 255), 3
                )  # unión de punto xy1 con xy2
                puntos = 2
                puntuacion.append(puntos)
            if sum(puntuacion) >= 2:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    img=manual_pose.image,
                    text="PARALLEL POSITION",
                    org=(int(width * 0.68), 30),
                    fontFace=font,
                    fontScale=0.7,
                    # text_color_bg=(127, 233, 100),
                    color=(255, 255, 230),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )
            # image = cv2.resize(src = uu.image, dsize = (616,1096))
            # image_rgb = cv2.cvtColor(manual_pose, cv2.COLOR_BGR2RGB)
            # Convert the BGR image to RGB and display it in Streamlit
            # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(manual_pose.image)
    else:
        st.write("No image captured. Please capture an image to proceed.")


##############################################################################
if option == "Upload a photo":
    st.subheader("Upload a photo")
    st.write(
        "Once you have uploaded a photo you will get the instant feedback about your pose "
    )
    st.write(
        "Even if this process isn’t in real-time, you’ll earn a point if your photo shows that you’ve gone below the parallel position in your squat."
    )
    uploaded_file = st.file_uploader("Choose a photo", type=["jpg", "png"])
    manual_pose = Manual_Pose_with_array()
    if uploaded_file is not None:
        # Convert the image to a numpy array
        np_image = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        # image = cv2.resize(src=image, dsize=(642, 1141))
        height, width, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        manual_pose.process(image_rgb)
        if manual_pose.pose_landmarks is not None:
            x0 = int(manual_pose.pose_landmarks["x"][0])
            y0 = int(manual_pose.pose_landmarks["y"][0])
            x1 = int(manual_pose.pose_landmarks["x"][1])
            y1 = int(manual_pose.pose_landmarks["y"][1])
            x2 = int(manual_pose.pose_landmarks["x"][2])
            y2 = int(manual_pose.pose_landmarks["y"][2])
            x3 = int(manual_pose.pose_landmarks["x"][3])
            y3 = int(manual_pose.pose_landmarks["y"][3])
            x4 = int(manual_pose.pose_landmarks["x"][4])
            y4 = int(manual_pose.pose_landmarks["y"][4])
            x5 = int(manual_pose.pose_landmarks["x"][5])
            y5 = int(manual_pose.pose_landmarks["y"][5])
            x6 = int(manual_pose.pose_landmarks["x"][6])
            y6 = int(manual_pose.pose_landmarks["y"][6])
            x7 = int(manual_pose.pose_landmarks["x"][7])
            y7 = int(manual_pose.pose_landmarks["y"][7])
            x8 = int(manual_pose.pose_landmarks["x"][8])
            y8 = int(manual_pose.pose_landmarks["y"][8])
            x9 = int(manual_pose.pose_landmarks["x"][9])
            y9 = int(manual_pose.pose_landmarks["y"][9])
            cv2.line(manual_pose.image, (x0, y0), (x3, y3), (255, 255, 255), 3)
            cv2.line(manual_pose.image, (x1, y1), (x3, y3), (255, 255, 255), 3)
            cv2.line(manual_pose.image, (x2, y2), (x3, y3), (255, 255, 255), 3)
            cv2.line(manual_pose.image, (x4, y4), (x5, y5), (255, 255, 255), 3)
            cv2.line(manual_pose.image, (x4, y4), (x0, y0), (255, 255, 255), 3)
            cv2.line(manual_pose.image, (x5, y5), (x1, y1), (255, 255, 255), 3)
            cv2.line(manual_pose.image, (x4, y4), (x6, y6), (255, 255, 255), 3)
            cv2.line(manual_pose.image, (x5, y5), (x7, y7), (255, 255, 255), 3)
            cv2.line(manual_pose.image, (x6, y6), (x8, y8), (255, 255, 255), 3)
            cv2.line(manual_pose.image, (x7, y7), (x9, y9), (255, 255, 255), 3)
            puntuacion = []
            if y4 >= y6 - 30 and y4 <= y6 + 30 or y5 >= y7 - 30 and y5 <= y7 + 30:
                cv2.line(
                    manual_pose.image, (x4, y4), (x6, y6), (0, 255, 255), 3
                )  # unión de punto xy1 con xy2
                cv2.line(
                    manual_pose.image, (x5, y5), (x7, y7), (0, 255, 255), 3
                )  # unión de punto xy1 con xy2
                puntos = 2
                puntuacion.append(puntos)
            if sum(puntuacion) >= 2:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    img=manual_pose.image,
                    text="PARALLEL POSITION",
                    org=(int(width * 0.68), 30),
                    fontFace=font,
                    fontScale=0.7,
                    # text_color_bg=(127, 233, 100),
                    color=(255, 255, 230),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )
            # image = cv2.resize(src = uu.image, dsize = (616,1096))
            # image_rgb = cv2.cvtColor(manual_pose, cv2.COLOR_BGR2RGB)
            # Convert the BGR image to RGB and display it in Streamlit
            # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(manual_pose.image)
    else:
        st.write("No image added. Please add an image to proceed.")

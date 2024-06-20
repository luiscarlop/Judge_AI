import streamlit as st
import base64
from PIL import Image
import streamlit as st
import cv2
import numpy as np
import av
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

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


# def main_page():
#     st.markdown("# 🏠Home")
#     st.sidebar.markdown("# 🏠Home")


# def page2():
#     st.markdown("# 🕺Profile")
#     st.sidebar.markdown("#🕺Profile")


# def page3():
#     st.markdown("# ⚖️Judge AI App")
#     st.sidebar.markdown("#⚖️Judge AI App")


# def page4():
#     st.markdown("# 🕵️‍♂️About_us")
#     st.sidebar.markdown("# 🕵️‍♂️About_us")


# page_names_to_funcs = {
#     "🏠Home": main_page,
#     "🕺Profile": page2,
#     "⚖️Judge AI App": page3,
#     "🕵️‍♂️About_us": page4,
# }

# selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
# page_names_to_funcs[selected_page]()

option = st.sidebar.selectbox(
    "Select an option", ("Intoduction", "Video", "Photo mode", "Upload a photo"), key=1
)
if option == "Intoduction":
    st.title("Welcome to the fun part")
    st.write(
        "If you are ready to start your workout, and you have your camera set up, you can select ether Video or Photo Mode."
    )
    st.write("In case you have a photo you can use the uploade freature.")
    st.write("Let's go!")

option = st.selectbox(
    "Select an option", ("Intoduction", "Video", "Photo mode", "Upload a photo")
)
if option == "Video":
    ####################################################################################### trasformador de video bueno
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    class VideoProcessor(VideoTransformerBase):
        def __init__(self):
            self.pose = mp_pose.Pose(static_image_mode=False)

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")

            height, width, _ = img.shape
            # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.pose.process(img)

            if results.pose_landmarks is not None:
                mp_drawing.draw_landmarks(
                    img,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(
                        color=(43, 214, 38), thickness=2, circle_radius=3
                    ),  # personalizar el esqueleto
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2),
                )

            return img

    webrtc_streamer(key="example", video_transformer_factory=VideoProcessor)


###################################################################################################

if option == "Photo mode":
    img_file_buffer = st.camera_input("Take a picture")
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    if img_file_buffer is not None:
        # Convert the image to a numpy array
        np_image = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)

        with mp_pose.Pose(
            static_image_mode=True
        ) as pose:  # PARA VIDEOS PONER FALSE /TRUE ES PARA FOTOS
            image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
            # image = cv2.resize(src=image, dsize=(642, 1141))
            # height, width, _ = image.shape
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = pose.process(image_rgb)

            if results.pose_landmarks is not None:
                # ... (your landmark drawing code here)

                mp_drawing.draw_landmarks(
                    image_rgb,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(
                        color=(43, 214, 38), thickness=2, circle_radius=3
                    ),  # personalizar el esqueleto
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2),
                )

            # Convert the BGR image to RGB and display it in Streamlit
            # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(image_rgb)
    else:
        st.write("No image captured. Please capture an image to proceed.")


if option == "Upload a photo":
    uploaded_file = st.file_uploader("Choose a photo", type=["jpg", "png"])
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.")
        # Convert the image to a numpy array
        np_image = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

        with mp_pose.Pose(
            static_image_mode=True
        ) as pose:  # PARA VIDEOS PONER FALSE /TRUE ES PARA FOTOS
            image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
            # image = cv2.resize(src=image, dsize=(642, 1141))
            height, width, _ = image.shape
            # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = pose.process(image)

            if results.pose_landmarks is not None:
                # ... (your landmark drawing code here)

                mp_drawing.draw_landmarks(
                    image_rgb,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(
                        color=(43, 214, 38), thickness=2, circle_radius=3
                    ),  # personalizar el esqueleto
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2),
                )

            # Convert the BGR image to RGB and display it in Streamlit
            # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(image_rgb)
    else:
        st.write("No image captured. Please capture an image to proceed.")

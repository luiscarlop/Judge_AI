
import streamlit as st
import base64
from PIL import Image
import streamlit as st
import cv2
import numpy as np
import av
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import io

from utils import draw_text
from VideoProcessor import VideoProcessor, PhotoProcessor
from FSM_squat import FSM_squat
from thresholds import get_thresholds


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

st.write("In this page we are using a pretrained mode")
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
    st.sidebar.markdown("#🕺Our model")


def page3():
    # st.markdown("#⚖️Pre-trained model")
    st.sidebar.markdown("# ⚖️Pre-trained model")


def page4():
    # st.markdown("# 🕵️‍♂️About_us")
    st.sidebar.markdown("#  🕵️‍♂️About_us")


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


if option == "Live video":
    #######################################################################################
    st.subheader("Live video")
    st.write("Click on the start button and start doing squats.")
    st.write("When you squat low enough, an indicator will appear on the screen")
    st.write(
        "Throughout the process, you will receive outputs indicating the state of your squat. \
    Once you finish your squat, it will be added to your score."
    )
    # mp_drawing = mp.solutions.drawing_utils
    # mp_pose = mp.solutions.pose

    webrtc_streamer(
        key="squat_judge",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )


###################################################################################################

if option == "Photo mode":

    st.subheader("Photo mode")
    st.write(
        "Take a photo when you’re at the lowest point of your squat. Please ask someone at the gym for help to avoid injuries!"
    )
    st.write("Oncet the photo is takes you can check how good you were doing it.")
    st.write(
        "Even if this process isn’t in real-time, you’ll earn a point if your photo shows that you’ve gone bellow the parallel position in your squat."
    )
    img_file_buffer = st.camera_input("Take a picture!")
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    PhotoProcessor = PhotoProcessor()

    if img_file_buffer is not None:
        # Convert the image to a numpy array
        np_image = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)

        image_rgb = PhotoProcessor.process(np_image)

        st.image(image_rgb)
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
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    PhotoProcessor = PhotoProcessor()

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(file_bytes))
        # st.image(image, caption="Uploaded Image.")
        np_image = np.asarray(bytearray(file_bytes), dtype=np.uint8)

        
        image_rgb = PhotoProcessor.process(np_image)
        # Convert the BGR image to RGB and display it in Streamlit
        # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb)
    else:
        st.write("No image added. Please add an image to proceed.")

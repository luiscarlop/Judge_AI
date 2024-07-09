from VideoProcessor import PhotoProcessor
import streamlit as st
import base64
from PIL import Image
import mediapipe as mp
import numpy as np
import cv2
import os
import moviepy.editor as mop


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
    "🕺Our model.py": page2,
    "⚖️Pre-trained model": page3,
    "🕵️‍♂️About_us": page4,
}

# selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
# page_names_to_funcs[selected_page]()

# st.title("This is Judge AI")
st.markdown(
    "<h1 style='text-align: center; '>This is Judge AI</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<hr style='border:2px solid #f3b323; margin-bottom: 0px; margin-top: 0px'>",
    unsafe_allow_html=True,
)

st.markdown(
    "<h6 style='text-align: center; '>This AI is designed to determine whether a user is performing their squats correctly. It counts the squats in real-time, \
    leveraging the power of our trained AI. .</h6>",
    unsafe_allow_html=True,
)

# st.markdown(
#     "<h2 style='text-align: center; '></h2>",
#     unsafe_allow_html=True,
# )

st.markdown("<h1 style='text-align: center;'>Uses</h1>", unsafe_allow_html=True)


# st.header(
#     "Uses",
#     divider="orange",
# )

st.markdown(
    "<h6 style='text-align: center; '>Experience the future of fitness with our cutting-edge tool, designed to revolutionize your daily workouts. With a focus on squats, our innovative solution automates the judging process in squat competitions, ensuring precision and fairness. Elevate your training regime and embrace the new standard in fitness technology .</h6>",
    unsafe_allow_html=True,
)


################################################################

# mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

IMG_WIDTH = 650
IMG_HEIGHT = 1141

img_paths = [
    "data/dataset_train/IMG_5615_018 (1).jpg",
    "data/dataset_train/IMG_5615_012.jpg",
]


image = cv2.imread(img_paths[0])
image_rgb = PhotoProcessor().process(image)
resized_image = cv2.resize(src=image_rgb, dsize=(IMG_WIDTH, IMG_HEIGHT))


col1, col2, col3, col4 = st.columns([1, 1, 6, 1])
with col3:
    st.image(
        resized_image,
        caption="An example of what the AI can do with a picture without breaking the pararel",
    )


#########################################################


image = cv2.imread(img_paths[1])
image_rgb = PhotoProcessor().process(image)
resized_image_2 = cv2.resize(src=image_rgb, dsize=(IMG_WIDTH, IMG_HEIGHT))


col1, col2, col3, col4 = st.columns([1, 1, 6, 1])

with col3:
    st.image(
        resized_image_2,
        caption="Breaking the paralel marks it and gives feedback to the user",
    )


###############################################################
input_video_path = "data/video_sample/IMG_5693.mov"
output_video_path = "data/video_sample/IMG_5693.mp4"


if not os.path.exists(output_video_path):
    PhotoProcessor().process(static_image_mode=False, video_path=input_video_path)

    # video = video_clip.set_audio(audio)
    # Write out the final video with audio
    # video.write_videofile(
    #     output_video_path, codec="ffv1", audio_codec="aac", bitrate="5000k"
    # )


(
    col1,
    col2,
    col3,
) = st.columns([1, 5, 1])

with col2:
    if not os.path.exists(output_video_path):
        st.video(
            PhotoProcessor().process(static_image_mode=False, video_path=input_video_path),
            format="H264",
            loop=True,
            autoplay=True,
            muted=True,
        )
        st.markdown(
            """<h6 style='text-align: center; '>Live and video tracking</h6>""",
            unsafe_allow_html=True,
        )
        
    else:
        st.video(
            output_video_path,
            format="H264",
            loop=True,
            autoplay=True,
            muted=True,
        )
        st.markdown(
            """<h6 style='text-align: center; '>Live and video tracking</h6>""",
            unsafe_allow_html=True,
        )
# st.video("data\video_sample\IMG_5619.mp4")

# st.subheader("This is a layer 2 text", divider=True)
# st.markdown("Cool")


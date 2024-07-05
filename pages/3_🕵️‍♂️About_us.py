import os
import streamlit as st
import cv2
import mediapipe as mp
from PIL import Image, ImageDraw
import numpy as np
import cv2
import base64

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
    # st.markdown("# 🕺Our model")
    st.sidebar.markdown("# 🕺Our model")


def page3():
    # st.markdown("# ⚖️Pre-trained model")
    st.sidebar.markdown("# ⚖️Pre-trained model")


def page4():
    # st.markdown("# 🕵️‍♂️About us")
    st.sidebar.markdown("# 🕵️‍♂️About us")


page_names_to_funcs = {
    "🏠Home": main_page,
    "🕺Our model": page2,
    "⚖️Pre-trained model": page3,
    "🕵️‍♂️About us": page4,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()

st.markdown(
    "<h1 style='text-align: center; '>Our mission AI</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<hr style='border:2px solid #f3b323; margin-bottom: 0px; margin-top: 0px'>",
    unsafe_allow_html=True,
)
# st.header("Our mission")

st.markdown(
    "<h2 style='text-align: center; '>The origins of Judge AI</h2>",
    unsafe_allow_html=True,
)

st.markdown(
    """<h7 style='text-align: center; '>Introducing our innovative fitness application, Judge AI. This cutting-edge tool is designed to revolutionize your workout routine by providing real-time feedback on your physical training.  
In its current initial phase, Judge AI focuses on perfecting one of the fundamental exercises - the squat. Our advanced technology analyzes your form and awards you points for each successful squat, 
turning your workout into a rewarding experience. But that’s just the beginning! We have ambitious plans for the future. We’re developing a competitive section to bring a new level of excitement and motivation to your fitness journey. 
Imagine challenging yourself against friends, family, or even global users - the possibilities are endless!
Moreover, we’re working on a feature to monitor your daily performance, providing you with valuable insights to help you understand your progress 
and achieve your fitness goals faster.
Join us on this exciting journey and let Judge AI be your personal digital fitness coach!</h7>""",
    unsafe_allow_html=True,
)


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
paths = [
    "data/dataset_train/IMG_5687(2)_008.jpg",
    "data/dataset_train/IMG_5693(1)_005.jpg",
]
processed_images = []


# image = cv2.imread("data/dataset_train/IMG_5687_015.jpg")


# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
for path in paths:
    image = cv2.imread(path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(
        static_image_mode=True
    ) as pose:  # PARA VIDEOS PONER FALSE /TRUE ES PARA FOTOS
        # image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        # image = cv2.resize(src=image, dsize=(642, 1141))
        height, width, _ = image_rgb.shape
        # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
                y24 >= y26 - 30
                and y24 <= y26 + 30
                and y23 >= y25 - 30
                and y23 <= y25 + 30
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

                    cv2.putText(
                        img=image,
                        text="GOOD JOB",
                        org=(200, 200),
                        fontFace=font,
                        fontScale=2,
                        color=(255, 255, 255),
                        thickness=2,
                        lineType=cv2.LINE_AA,
                    )
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            processed_images.append(image_rgb)

print(image_rgb.shape)
new_width = 235
new_height = 410
resized_image = cv2.resize(src=image_rgb, dsize=(new_width, new_height))
print(resized_image.shape)

col1, col2, col3, col4 = st.columns([1.5, 1, 1, 1.5])
with col1:
    st.image(processed_images[0], caption="Mark pose created with our model")
with col4:
    st.image(processed_images[1], caption="Mark pose created with a pretrained model")


st.markdown(
    "<h1 style='text-align: center; '>Our next steps</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<hr style='border:1px solid #f3b323; margin-bottom: 0px; margin-top: 0px'>",
    unsafe_allow_html=True,
)


st.markdown(
    "<h2 style='text-align: center; '>The Future of Judge AI</h2>",
    unsafe_allow_html=True,
)


st.markdown(
    """<h7 style='text-align: center; '>As we continue to innovate and expand, Judge AI is set to redefine the landscape of fitness apps. 
Our future plans include the development of a robust competition system connected through the tournament mode and will serve as the judge,
ensuring a fair and objective competition.
This feature will match you with competitors who have similar personal records in the exercises you're competing in, 
adding a thrilling dimension to your fitness journey.
But that's not all! We're also integrating a comprehensive social platform into Judge AI. 
This will include forums for discussion, clubs for shared interests, and even the option to hire a professional coach. 
A friend system will also be in place, fostering a supportive and motivating fitness community.
One of the standout features we're excited about is our competition system where Judge AI itself will be the judge. 
We are introducing two types of competitions: matchmaking competitions and official competitions. 
In matchmaking competitions, competitors can compete online from anywhere. 
In official competitions, such as a powerlifter squat tournament, competitors will gather at a physical location. 
Our AI will include 1 vs 1 mode  that will allow you to compete online with a refined matchmaking considering all the stored data. 
Your personal profile will be a hub of information. 
It will display your all-time stats in a visually appealing and easy-to-understand format. 
Want to see how you stack up against your friends? You'll have the option to compare your stats with theirs, adding a friendly competitive edge
to your workout routine.
Join us as we take fitness to the next level with Judge AI, your ultimate fitness companion!</h7>""",
    unsafe_allow_html=True,
)


st.markdown(
    "<h1 style='text-align: center; '>Our team</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<hr style='border:1px solid #f3b323; margin-bottom: 0px; margin-top: 0px'>",
    unsafe_allow_html=True,
)


# st.markdown(
#     "<h2 style='text-align: left; '>Luis Carretero</h2>",
#     unsafe_allow_html=True,
# )
with open(
    "data/stored_pictures/linkedin-logo-free-download-free-png.webp", "rb"
) as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()

with open("data/stored_pictures/github_PNG40.png", "rb") as image_file:
    encoded_string2 = base64.b64encode(image_file.read()).decode()

st.markdown(
    f"""
    <div style="display: flex; align-items: center; justify-content: flex-start;">
        <h2 style='margin-right: 20px;'>Luis Carretero</h2>
        <a href="https://www.linkedin.com/in/claudio-desco-serrano-b787991a8/">
            <img src="data:image/png;base64,{encoded_string}" width="60">
        </a>
        <a href="https://github.com/luiscarlop">
            <img src="data:image/png;base64,{encoded_string2}" width="40" style='margin-left: 20px;'>
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)


luis_path = "data/stored_pictures/photo.jpg"
luis_image = cv2.imread(luis_path)
lu_rgb = cv2.cvtColor(luis_image, cv2.COLOR_BGR2RGB)
new_width = 265
new_height = 410
resized_image = cv2.resize(src=lu_rgb, dsize=(new_width, new_height))
col5, col6, col7, col8 = st.columns([1.5, 10, 1, 1.5])
with col5:
    st.image(
        resized_image,
    )
with col6:
    st.markdown(
        """<h7 style='text-align: center; '>Hola caracola tu texto aqui</h7>""",
        unsafe_allow_html=True,
    )

col9, col10 = st.columns([1, 1])
with col9:
    st.markdown(":e-mail: luiscarretero.tech@gmail.com")


with open(
    "data/stored_pictures/linkedin-logo-free-download-free-png.webp", "rb"
) as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()


st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <h2 style='margin-right: 20px;'>Claudio Desco</h2>
        <a href="https://www.linkedin.com/in/claudio-desco-serrano-b787991a8/">
            <img src="data:image/png;base64,{encoded_string}" width="60">
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)

claudio_path = "data/stored_pictures/photo.jpg"
cl_image = cv2.imread(claudio_path)
cl_rgb = cv2.cvtColor(cl_image, cv2.COLOR_BGR2RGB)
new_width = 265
new_height = 410
resized_image = cv2.resize(src=cl_rgb, dsize=(new_width, new_height))
col5, col6, col7, col8 = st.columns([1.5, 10, 1, 1.5])
with col5:
    st.image(
        resized_image,
    )

with col6:
    st.markdown(
        """<h7 style='text-align: center; '>Claudio Desco, is a multifaceted professional with equal expertise in Marketing, Tourism, and Data Science. Claudio’s strong marketing background, coupled with his degree in Tourism and a bootcamp in Data Science, allows him to craft strategies that are both innovative and globally relevant
        At Judge AI, he applies his diverse skill set, using data-driven insights to inform marketing decisions and artificial intelligence to optimize outcomes. His work is characterized by a comprehensive understanding of global trends, a knack for extracting meaningful insights from data, and the ability to translate these into effective marketing strategies.
        In summary, Claudio Desco exemplifies the potency of cross-disciplinary proficiency. His distinctive amalgamation of expertise in Marketing, Tourism, and Data Science, coupled with his unwavering dedication to innovation at Judge AI, places him as a leading figure in his domain.</h7>""",
        unsafe_allow_html=True,
    )

col9, col10 = st.columns([1, 1])
with col9:
    st.markdown(":e-mail: claudiodescopersonal@gmail.com")
# st.image(resized_image)

# st.markdown(
#     "<h2 style='text-align: left; '>Noelia Vergara</h2>",
#     unsafe_allow_html=True,
# )
with open(
    "data/stored_pictures/linkedin-logo-free-download-free-png.webp", "rb"
) as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()


st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <h2 style='margin-right: 20px;'>Noelia Vergara</h2>
        <a href="https://www.linkedin.com/in/claudio-desco-serrano-b787991a8/">
            <img src="data:image/png;base64,{encoded_string}" width="60">
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)


noe_path = "data/stored_pictures/photo.jpg"
noe_image = cv2.imread(noe_path)
noe_rgb = cv2.cvtColor(noe_image, cv2.COLOR_BGR2RGB)
new_width = 265
new_height = 410
resized_image = cv2.resize(src=noe_rgb, dsize=(new_width, new_height))
col5, col6, col7, col8 = st.columns([1.5, 10, 1, 1.5])
with col5:
    st.image(
        resized_image,
    )

with col6:
    st.markdown(
        """<h7 style='text-align: center; '>Hola caracola aqui tu texto</h7>""",
        unsafe_allow_html=True,
    )

col9, col10 = st.columns([1, 1])
with col9:
    st.markdown(":e-mail: tuemail@mail.com")

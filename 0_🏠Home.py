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


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


image = cv2.imread("data/dataset_train/IMG_5615_018 (1).jpg")


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

        if y24 >= y26 - 30 and y24 <= y26 + 30 and y23 >= y25 - 30 and y23 <= y25 + 30:
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

print(image_rgb.shape)
new_width = 650
new_height = 1141
resized_image = cv2.resize(src=image_rgb, dsize=(new_width, new_height))
print(resized_image.shape)

col1, col2, col3, col4 = st.columns([1, 1, 6, 1])
with col3:
    st.image(
        resized_image,
        caption="An example of what the AI can do with a picture without breaking the pararel",
    )


#########################################################


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


image = cv2.imread("data/dataset_train/IMG_5615_012.jpg")


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

        if y24 >= y26 - 30 and y24 <= y26 + 30 and y23 >= y25 - 30 and y23 <= y25 + 30:
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

print(image_rgb.shape)
new_width = 650
new_height = 1141
resized_image_2 = cv2.resize(src=image_rgb, dsize=(new_width, new_height))
print(resized_image.shape)

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

    cap = cv2.VideoCapture(input_video_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"H264")
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (frame_width, frame_height))

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    with mp_pose.Pose(static_image_mode=False) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            height, width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks is not None:
                print(
                    int(
                        results.pose_landmarks.landmark[
                            mp_pose.PoseLandmark.RIGHT_SHOULDER
                        ].z
                    )
                )  # saca la coordenada del hombro derecho, medida en cm se multiplica por el ancho para sacar la medida en pixeles
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

                print(x24, y24, x26, y26, x23, y23, x25, y25, x27, y27)
                cv2.line(
                    frame, (x11, y11), (x12, y12), (255, 255, 255), 3
                )  # unión de punto xy1 con xy2
                cv2.line(frame, (x11, y11), (x13, y13), (255, 255, 255), 3)
                cv2.line(
                    frame, (x11, y11), (x23, y23), (255, 255, 255), 3
                )  # unión de punto xy2 con xy3
                cv2.line(frame, (x13, y13), (x15, y15), (255, 255, 255), 3)
                cv2.line(frame, (x12, y12), (x14, y14), (255, 255, 255), 3)
                cv2.line(frame, (x14, y14), (x16, y16), (255, 255, 255), 3)
                cv2.line(frame, (x12, y12), (x24, y24), (255, 255, 255), 3)
                cv2.line(frame, (x24, y24), (x26, y26), (255, 255, 255), 3)
                cv2.line(frame, (x24, y24), (x23, y23), (255, 255, 255), 3)
                cv2.line(frame, (x26, y26), (x28, y28), (255, 255, 255), 3)
                cv2.line(frame, (x23, y23), (x25, y25), (255, 255, 255), 3)
                cv2.line(frame, (x25, y25), (x27, y27), (255, 255, 255), 3)

                cv2.circle(frame, (x11, y11), 6, (43, 214, 38), -1)  # color verde
                cv2.circle(frame, (x12, y12), 6, (43, 214, 38), -1)  # color verde
                cv2.circle(frame, (x13, y13), 6, (43, 214, 38), -1)  # color verde
                cv2.circle(frame, (x15, y15), 6, (43, 214, 38), -1)  # color verde
                cv2.circle(frame, (x14, y14), 6, (43, 214, 38), -1)  # color verde
                cv2.circle(frame, (x16, y16), 6, (43, 214, 38), -1)  # color verde
                cv2.circle(frame, (x24, y24), 6, (43, 214, 38), -1)  # color verde
                cv2.circle(frame, (x26, y26), 6, (43, 214, 38), -1)  # color verde
                cv2.circle(frame, (x28, y28), 6, (43, 214, 38), -1)  # color verde
                cv2.circle(
                    frame, (x23, y23), 6, (43, 214, 38), -1
                )  # color azul corrdenadas según indice del dibujo
                cv2.circle(
                    frame, (x25, y25), 6, (43, 214, 38), -1
                )  # color azul corrdenadas según indice del dibujo
                cv2.circle(frame, (x27, y27), 6, (43, 214, 38), -1)

                if (
                    y24 >= y26 - 30
                    and y24 <= y26 + 30
                    and y23 >= y25 - 30
                    and y23 <= y25 + 30
                ):
                    cv2.line(
                        frame, (x24, y24), (x26, y26), (0, 255, 255), 3
                    )  # unión de punto xy1 con xy2
                    cv2.line(
                        frame, (x23, y23), (x25, y25), (0, 255, 255), 3
                    )  # unión de punto xy1 con xy2

                    puntos = 2
                    puntuacion.append(puntos)
                    print(f" puntuación: {sum(puntuacion)} ")

                if sum(puntuacion) >= 2:

                    font = cv2.FONT_HERSHEY_SIMPLEX

                    cv2.putText(
                        img=frame,
                        text="GOOD JOB",
                        org=(200, 200),
                        fontFace=font,
                        fontScale=2,
                        color=(255, 255, 255),
                        thickness=2,
                        lineType=cv2.LINE_AA,
                    )
            out.write(frame)
        cap.release()
        out.release()  # Write out frame to video

    cap.release()
    out.release()
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

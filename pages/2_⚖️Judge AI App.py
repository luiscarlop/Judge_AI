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

st.title("Welcome to the fun part")
st.write(
    "If you are ready to start your workout, and you have your camera set up, you can select ether Video or Photo Mode."
)
st.write("In case you have a photo you can use the uploade freature.")
st.write("Let's go!")

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

option = st.selectbox(
    "Select an option",
    ("Intoduction", "Video of your camera", "Photo mode", "Upload a photo"),
    key=1,
)
if option == "Intoduction":
    st.subheader("We have 3 modes of use yet")
    st.write(
        "If you are ready to start your workout, and you have your camera set up, you can select ether Video or Photo Mode."
    )
    st.write("In case you have a photo you can use the uploade freature.")
    st.write("Let's go!")

option = st.selectbox(
    "Select an option",
    ("Intoduction", "Video of your camera", "Photo mode", "Upload a photo"),
)
if option == "Video of your camera":
    ####################################################################################### trasformador de video bueno
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    class VideoProcessor(VideoTransformerBase):
        def __init__(self):
            self.pose = mp_pose.Pose(static_image_mode=False)

        def transform(self, frame):
            frame = frame.to_ndarray(format="bgr24")

            height, width, _ = frame.shape
            # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame)

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

            return frame

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
        file_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(file_bytes))
        # st.image(image, caption="Uploaded Image.")
        np_image = np.asarray(bytearray(file_bytes), dtype=np.uint8)

        with mp_pose.Pose(
            static_image_mode=True
        ) as pose:  # PARA VIDEOS PONER FALSE /TRUE ES PARA FOTOS
            image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
            # image = cv2.resize(src=image, dsize=(642, 1141))
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
                # Convert the BGR image to RGB and display it in Streamlit
                # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                st.image(image_rgb)
    else:
        st.write("No image added. Please add an image to proceed.")

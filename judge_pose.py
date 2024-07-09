from utils import get_manual_landmark_features

from typing import Generator, Iterator, Tuple
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import xml.etree.ElementTree as ET
from flow_from_directory_noelia import restaurar_imagen, crear_pares
import cv2
from sklearn.model_selection import train_test_split
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp

DATA_DIR = Path("data")
IMG_SIZE = 224
NUM_POINTS = 10


class Manual_Pose_with_array:
    def __init__(self):
        self.dict_features = {
            "right_shoulder": 0,
            "left_shoulder": 1,
            "head": 2,
            "collarbone": 3,
            "right_hip": 4,
            "left_hip": 5,
            "right_knee": 6,
            "left_knee": 7,
            "right_ankle": 8,
            "left_ankle": 9,
        }

    def process(self, array):
        IMG_SIZE = 224

        # imagen = cv2.imread(path)

        image_array = np.asarray(array)

        imagen_redimensionada = Image.fromarray(image_array)
        imagen_redimensionada.save("imagen_redimensionada.jpg")

        # Devolver la imagen a su tamaño original
        imagen_restaurada = imagen_redimensionada.resize((IMG_SIZE, IMG_SIZE))
        # imagen_restaurada = cv2.cvtColor(imagen_restaurada, cv2.COLOR_BGR2RGB)
        imagen_restaurada.save("imagen_restaurada.jpg")

        imagen_bgr = cv2.imread("imagen_restaurada.jpg")

        # Convertir de BGR a RGB usando NumPy
        imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)

        # Guardar la imagen convertida a RGB
        cv2.imwrite("imagen_restaurada.jpg", imagen_rgb)

        new_array = np.array(np.vstack(imagen_rgb))

        new_array = new_array.reshape(1, IMG_SIZE, IMG_SIZE, 3)
        new_array = new_array / 255

        model = load_model("judge_best_model_reentrenado.keras")

        new_yhat = model.predict(new_array)

        imagen_redimensionada2 = Image.fromarray(image_array)
        imagen_redimensionada2.save("imagen_redimensionada2.jpg")

        # Devolver la imagen a su tamaño original
        imagen_restaurada2 = imagen_redimensionada.resize((1080, 1920))
        # imagen_restaurada = cv2.cvtColor(imagen_restaurada, cv2.COLOR_BGR2RGB)
        imagen_restaurada2.save("imagen_restaurada2.jpg")

        imagen_bgr2 = cv2.imread("imagen_restaurada2.jpg")

        # Convertir de BGR a RGB usando NumPy
        imagen_rgb2 = cv2.cvtColor(imagen_bgr2, cv2.COLOR_BGR2RGB)

        # Guardar la imagen convertida a RGB

        cv2.imwrite("imagen_restaurada2.jpg", imagen_rgb2)

        self.image = imagen_rgb2

        keypoints = crear_pares(new_yhat[0])

        for x, y in keypoints:

            cv2.circle(
                img=imagen_rgb2,
                center=(int(x), int(y)),
                radius=8,
                color=(255, 50, 50),
                thickness=-1,
            )

        plt.imshow(imagen_rgb2)
        plt.show()

        x_points = list()
        y_points = list()
        landmarks_df = pd.DataFrame()
        for x, y in keypoints:
            x = int(x)
            y = int(y)

            x_points.append(x)
            y_points.append(y)
            # landmarks["index"] = range(1,len(x_points))

        landmarks_df["x"] = x_points
        landmarks_df["y"] = y_points

        # features=["right_shoulder","left_shoulder","head","collarbone","right_hip","left_hip",
        # "right_knee","left_knee","right_ankle","left_ankle"]

        self.pose_landmarks = landmarks_df
        return self.pose_landmarks  # imagen_rgb2, keypoints ,

    def pose_landmarks(self):
        self.pose_landmarks


class Manual_Pose_with_path:

    def process(self, path):
        IMG_SIZE = 224

        imagen = cv2.imread(path)

        image_array = np.asarray(imagen)

        imagen_redimensionada = Image.fromarray(image_array)
        imagen_redimensionada.save("imagen_redimensionada.jpg")

        # Devolver la imagen a su tamaño original
        imagen_restaurada = imagen_redimensionada.resize((IMG_SIZE, IMG_SIZE))
        # imagen_restaurada = cv2.cvtColor(imagen_restaurada, cv2.COLOR_BGR2RGB)
        imagen_restaurada.save("imagen_restaurada.jpg")

        imagen_bgr = cv2.imread("imagen_restaurada.jpg")

        # Convertir de BGR a RGB usando NumPy
        imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)

        # Guardar la imagen convertida a RGB
        cv2.imwrite("imagen_restaurada.jpg", imagen_rgb)

        new_array = np.array(np.vstack(imagen_rgb))

        new_array = new_array.reshape(1, IMG_SIZE, IMG_SIZE, 3)
        new_array = new_array / 255

        model = load_model("judge_best_model_reentrenado.keras")

        new_yhat = model.predict(new_array)

        imagen_redimensionada2 = Image.fromarray(image_array)
        imagen_redimensionada2.save("imagen_redimensionada2.jpg")

        # Devolver la imagen a su tamaño original
        imagen_restaurada2 = imagen_redimensionada.resize((1080, 1920))
        # imagen_restaurada = cv2.cvtColor(imagen_restaurada, cv2.COLOR_BGR2RGB)
        imagen_restaurada2.save("imagen_restaurada2.jpg")

        imagen_bgr2 = cv2.imread("imagen_restaurada2.jpg")

        # Convertir de BGR a RGB usando NumPy
        imagen_rgb2 = cv2.cvtColor(imagen_bgr2, cv2.COLOR_BGR2RGB)

        # Guardar la imagen convertida a RGB

        cv2.imwrite("imagen_restaurada2.jpg", imagen_rgb2)

        self.image = imagen_rgb2

        keypoints = crear_pares(new_yhat[0])

        for x, y in keypoints:

            cv2.circle(
                img=imagen_rgb2,
                center=(int(x), int(y)),
                radius=8,
                color=(255, 50, 50),
                thickness=-1,
            )

        plt.imshow(imagen_rgb2)
        plt.show()

        x_points = list()
        y_points = list()
        landmarks_df = pd.DataFrame()
        for x, y in keypoints:
            x = int(x)
            y = int(y)

            x_points.append(x)
            y_points.append(y)
            # landmarks["index"] = range(1,len(x_points))

        landmarks_df["x"] = x_points
        landmarks_df["y"] = y_points

        # features=["right_shoulder","left_shoulder","head","collarbone","right_hip","left_hip",
        # "right_knee","left_knee","right_ankle","left_ankle"]

        self.pose_landmarks = landmarks_df
        return self.pose_landmarks  # imagen_rgb2, keypoints ,

    def pose_landmarks(self):
        self.pose_landmarks


class Manual_Pose_video:

    def process(self, path):
        IMG_SIZE = 224

        # imagen = cv2.imread(path)

        image_array = np.asarray(path)  # es imagen solo pruebo

        imagen_redimensionada = Image.fromarray(image_array)
        imagen_redimensionada.save("imagen_redimensionada.jpg")

        # Devolver la imagen a su tamaño original
        imagen_restaurada = imagen_redimensionada.resize((IMG_SIZE, IMG_SIZE))
        # imagen_restaurada = cv2.cvtColor(imagen_restaurada, cv2.COLOR_BGR2RGB)
        imagen_restaurada.save("imagen_restaurada.jpg")

        imagen_bgr = cv2.imread("imagen_restaurada.jpg")

        # Convertir de BGR a RGB usando NumPy
        imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)

        # Guardar la imagen convertida a RGB
        cv2.imwrite("imagen_restaurada.jpg", imagen_rgb)

        new_array = np.array(np.vstack(imagen_rgb))

        new_array = new_array.reshape(1, IMG_SIZE, IMG_SIZE, 3)
        new_array = new_array / 255

        model = load_model("judge_best_model_reentrenado.keras")

        new_yhat = model.predict(new_array)

        imagen_redimensionada2 = Image.fromarray(image_array)
        imagen_redimensionada2.save("imagen_redimensionada2.jpg")

        # Devolver la imagen a su tamaño original
        imagen_restaurada2 = imagen_redimensionada.resize((616, 1096))
        # imagen_restaurada = cv2.cvtColor(imagen_restaurada, cv2.COLOR_BGR2RGB)
        imagen_restaurada2.save("imagen_restaurada2.jpg")

        imagen_bgr2 = cv2.imread("imagen_restaurada2.jpg")

        # Convertir de BGR a RGB usando NumPy
        imagen_rgb2 = cv2.cvtColor(imagen_bgr2, cv2.COLOR_BGR2RGB)

        # Guardar la imagen convertida a RGB

        cv2.imwrite("imagen_restaurada2.jpg", imagen_rgb2)

        self.image = imagen_rgb2

        keypoints = crear_pares(new_yhat[0])

        for x, y in keypoints:

            cv2.circle(
                img=imagen_rgb2,
                center=(int(x), int(y)),
                radius=8,
                color=(255, 50, 50),
                thickness=-1,
            )

        plt.imshow(imagen_rgb2)
        plt.show()

        x_points = list()
        y_points = list()
        landmarks_df = pd.DataFrame()
        for x, y in keypoints:
            x = int(x)
            y = int(y)

            x_points.append(x)
            y_points.append(y)
            # landmarks["index"] = range(1,len(x_points))

        landmarks_df["x"] = x_points
        landmarks_df["y"] = y_points

        # features=["right_shoulder","left_shoulder","head","collarbone","right_hip","left_hip",
        # "right_knee","left_knee","right_ankle","left_ankle"]

        self.pose_landmarks = landmarks_df
        return self.pose_landmarks  # imagen_rgb2, keypoints ,

    def pose_landmarks(self):
        self.pose_landmarks

import os
import json

import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from PIL import Image
from pathlib import Path
from typing import Generator, Iterator, Tuple

DATA_DIR = Path('data')


def draw_rounded_rect(img, rect_start, rect_end, corner_width, box_color):

    x1, y1 = rect_start
    x2, y2 = rect_end
    w = corner_width

    # draw filled rectangles
    cv2.rectangle(img, (x1 + w, y1), (x2 - w, y1 + w), box_color, -1)
    cv2.rectangle(img, (x1 + w, y2 - w), (x2 - w, y2), box_color, -1)
    cv2.rectangle(img, (x1, y1 + w), (x1 + w, y2 - w), box_color, -1)
    cv2.rectangle(img, (x2 - w, y1 + w), (x2, y2 - w), box_color, -1)
    cv2.rectangle(img, (x1 + w, y1 + w), (x2 - w, y2 - w), box_color, -1)


    # draw filled ellipses
    cv2.ellipse(img, (x1 + w, y1 + w), (w, w),
                angle = 0, startAngle = -90, endAngle = -180, color = box_color, thickness = -1)

    cv2.ellipse(img, (x2 - w, y1 + w), (w, w),
                angle = 0, startAngle = 0, endAngle = -90, color = box_color, thickness = -1)

    cv2.ellipse(img, (x1 + w, y2 - w), (w, w),
                angle = 0, startAngle = 90, endAngle = 180, color = box_color, thickness = -1)

    cv2.ellipse(img, (x2 - w, y2 - w), (w, w),
                angle = 0, startAngle = 0, endAngle = 90, color = box_color, thickness = -1)

    return img 



def draw_dotted_line(frame, lm_coord, start, end, line_color):
    pix_step = 0

    for i in range(start, end+1, 8):
        cv2.circle(frame, (lm_coord[0], i+pix_step), 2, line_color, -1, lineType=cv2.LINE_AA)

    return frame


def draw_text(
    img,
    msg,
    width = 8,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    pos=(0, 0),
    font_scale=1,
    font_thickness=2,
    text_color=(0, 255, 0),
    text_color_bg=(0, 0, 0),
    box_offset=(20, 10),
):

    offset = box_offset
    x, y = pos
    text_size, _ = cv2.getTextSize(msg, font, font_scale, font_thickness)
    text_w, text_h = text_size
    rec_start = tuple(p - o for p, o in zip(pos, offset))
    rec_end = tuple(m + n - o for m, n, o in zip((x + text_w, y + text_h), offset, (25, 0)))
    
    img = draw_rounded_rect(img, rec_start, rec_end, width, text_color_bg)


    cv2.putText(
        img,
        msg,
        (int(rec_start[0] + 6), int(y + text_h + font_scale - 1)), 
        font,
        font_scale,
        text_color,
        font_thickness,
        cv2.LINE_AA,
    )

    
    return text_size




def find_angle_abs(p1, p2, ref_pt = np.array([0,0])):
    """Find the angle between two points. Formula: 
    cos(theta) = (p1_ref . p2_ref) / (||p1_ref|| * ||p2_ref||)
    theta = arccos(cos(theta)) --> angle in radians

    Args:
        p1 (_type_): _description_
        p2 (_type_): _description_
        ref_pt (_type_, optional): _description_. Defaults to np.array([0,0]).

    Returns:
        _type_: _description_
    """
    p1_ref = p1 - ref_pt
    p2_ref = p2 - ref_pt

    cos_theta = (np.dot(p1_ref,p2_ref)) / (1.0 * np.linalg.norm(p1_ref) * np.linalg.norm(p2_ref))
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
            
    degree = int(180 / np.pi) * theta # convert radians to degrees

    return int(degree)


def find_angle(p1, p2, ref_pt=np.array([0, 0])):
    """Find the angle between two points with respect to a vertical reference.
    Args:
        p1 (np.array): First point as an array [x, y].
        p2 (np.array): Second point as an array [x, y].
        ref_pt (np.array, optional): Reference point. Defaults to np.array([0,0]).
    Returns:
        float: Angle in degrees between -180 and 180.
    """
    p1_ref = p1 - ref_pt
    p2_ref = p2 - ref_pt

    # Calculate angle with respect to the vertical reference
    angle_p1 = np.arctan2(p1_ref[1], p1_ref[0])
    angle_p2 = np.arctan2(p2_ref[1], p2_ref[0])

    # Calculate the difference between the two angles
    theta = np.degrees(angle_p2 - angle_p1)

    # Normalize the angle to the range -180 to 180
    if theta > 180:
        theta -= 360
    elif theta < -180:
        theta += 360

    return int(theta)



def get_landmark_array(pose_landmark, key, frame_width, frame_height, manual: bool = False):

    if not manual:
        denorm_x = int(pose_landmark[key].x * frame_width)
        denorm_y = int(pose_landmark[key].y * frame_height)
    else:
        denorm_x = int(pose_landmark["x"][key])
        denorm_y = int(pose_landmark["y"][key])

    return np.array([denorm_x, denorm_y])

def get_manual_landmark_features(pose_landmark, dict_features, frame_width, frame_height):
    
    right_shoulder  = get_landmark_array(pose_landmark, dict_features['right_shoulder'], frame_width, frame_height, manual=True)
    left_shoulder   = get_landmark_array(pose_landmark, dict_features['left_shoulder'], frame_width, frame_height, manual=True)
    head            = get_landmark_array(pose_landmark, dict_features['head'], frame_width, frame_height, manual=True)
    collarbone      = get_landmark_array(pose_landmark, dict_features['collarbone'], frame_width, frame_height, manual=True)
    right_hip       = get_landmark_array(pose_landmark, dict_features['right_hip'], frame_width, frame_height, manual=True)
    left_hip        = get_landmark_array(pose_landmark, dict_features['left_hip'], frame_width, frame_height, manual=True)
    right_knee      = get_landmark_array(pose_landmark, dict_features['right_knee'], frame_width, frame_height, manual=True)
    left_knee       = get_landmark_array(pose_landmark, dict_features['left_knee'], frame_width, frame_height, manual=True)
    right_ankle     = get_landmark_array(pose_landmark, dict_features['right_ankle'], frame_width, frame_height, manual=True)
    left_ankle      = get_landmark_array(pose_landmark, dict_features['left_ankle'], frame_width, frame_height, manual=True)

    return right_shoulder, left_shoulder, head, collarbone, right_hip, left_hip, right_knee, left_knee, right_ankle, left_ankle
    


def get_landmark_features(kp_results, dict_features, feature, frame_width, frame_height):

    if feature == 'nose':
        return get_landmark_array(kp_results, dict_features[feature], frame_width, frame_height)

    elif feature == 'left' or 'right':
        shldr_coord = get_landmark_array(kp_results, dict_features[feature]['shoulder'], frame_width, frame_height)
        elbow_coord   = get_landmark_array(kp_results, dict_features[feature]['elbow'], frame_width, frame_height)
        wrist_coord   = get_landmark_array(kp_results, dict_features[feature]['wrist'], frame_width, frame_height)
        hip_coord   = get_landmark_array(kp_results, dict_features[feature]['hip'], frame_width, frame_height)
        knee_coord   = get_landmark_array(kp_results, dict_features[feature]['knee'], frame_width, frame_height)
        ankle_coord   = get_landmark_array(kp_results, dict_features[feature]['ankle'], frame_width, frame_height)
        foot_coord   = get_landmark_array(kp_results, dict_features[feature]['foot'], frame_width, frame_height)

        return shldr_coord, elbow_coord, wrist_coord, hip_coord, knee_coord, ankle_coord, foot_coord
    
    else:
        raise ValueError("feature needs to be either 'nose', 'left' or 'right")


def get_mediapipe_pose(
                        static_image_mode = False, 
                        model_complexity = 1,
                        smooth_landmarks = True,
                        min_detection_confidence = 0.5,
                        min_tracking_confidence = 0.5

                    ):
    pose = mp.solutions.pose.Pose(
                                    static_image_mode = static_image_mode,
                                    model_complexity = model_complexity,
                                    smooth_landmarks = smooth_landmarks,
                                    min_detection_confidence = min_detection_confidence,
                                    min_tracking_confidence = min_tracking_confidence
                                )
    return pose

def load_img(img_path: str) -> np.ndarray:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def parse_xml(file: str) -> ET.Element:
    """Parse an XML file and return the root element.

    Args:
        file (str): XML file to parse

    Returns:
        ET.Element: root element of the XML file
    """

    tree = ET.parse(file)
    root = tree.getroot()
    
    return root

def add_annotations_from_file(file: str, combined_root: ET.Element) -> ET.Element:
    """Add annotations from an XML file to a combined XML root.

    Args:
        file (str): XML file to add annotations from
        combined_root (ET.Element): XML root to add annotations to

    Returns:
        ET.Element: combined XML root with added annotations
    """
    root = parse_xml(file)
    
    for image in root.findall("image"):
        combined_root.append(image)



def pair_annotations_with_images(labels_dir: str) -> dict:
    """Pair image paths with their corresponding labels.

    Args:
        img_dir (str): directory containing images
        labels_dir (str): directory containing labels

    Returns:
        dict: dictionary of image paths and their corresponding labels
    """
    labels_root = parse_xml(labels_dir)

    annotations_dict = {}

    for image in labels_root.findall("image"):
        img_name = image.get("name")
        img = plt.imread(os.path.join(DATA_DIR / 'dataset_train', img_name))
        skeletons = []
        img_data = {}

        img_data['image_path'] = os.path.join(DATA_DIR / 'dataset_train', img_name)
        img_data['image_height'], img_data['image_width'] = img.shape[:2]
        
        for skeleton in image.findall("skeleton"):
            # skeleton_nodes = []
            for point in skeleton.findall("points"):
                x = point.get('points').split(',')[0]
                y = point.get('points').split(',')[1]
                # skeleton_nodes.append((x, y))
                skeletons.append([x, y])
            img_data['keypoints'] = skeletons
        annotations_dict[img_name] = img_data  

    return annotations_dict


def draw_keypoints(image: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    """Draw keypoints on an image.

    Args:
        image (np.ndarray): input image
        keypoints (np.ndarray): keypoints to draw

    Returns:
        np.ndarray: image with keypoints drawn
    """
    for x, y in keypoints:
        cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)
    
    # Mostrar la imagen con los puntos
    plt.imshow(image)

def image_scaler(image: np.ndarray, keypoints: np.ndarray, size: Tuple[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Scale keypoints to a new size.

    Args:
        keypoints (np.ndarray): keypoints to scale
        size (Tuple[int]): new size to scale keypoints to

    Returns:
        np.ndarray: scaled keypoints
    """
    h, w = image.shape[:2]
    new_h, new_w = size

    scaled_img = cv2.resize(image, (new_w, new_h))
    scaled_keypoints = keypoints * np.array([new_w / w, new_h / h])

    return scaled_img, scaled_keypoints


def read_json(json_path: str) -> dict:
    with open(json_path, 'r') as f:
        data = json.load(f)

    return data

def restore_image(image: np.ndarray, width, height):

    redim_img = Image.fromarray(image)
    # imagen_redimensionada.save("imagen_redimensionada.jpg")

    # Devolver la imagen a su tamaño original
    restored_img = redim_img.resize((width, height))

    # Convertir de BGR a RGB usando NumPy
    img_rgb = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
    
    return img_rgb  # se queda guardada en rgb

def pair_coords(coord_list):
    pares = []
    for i in range(0, len(coord_list), 2):
        if i + 1 < len(
            coord_list
        ):  # Asegurar que haya suficientes elementos para formar el par
            par = (coord_list[i], coord_list[i + 1])
            pares.append(par)

    return pares

def compare_keypoints(path, y, yhat):

    imagen = cv2.imread(path)

    pares_yhat = pair_coords(yhat)
    pares_y = pair_coords(y)

    for x, y in pares_yhat:

        cv2.circle(
            img=imagen,
            center=(int(x), int(y)),
            radius=13,
            color=(0, 255, 0),
            thickness=-1,
        )

    for x, y in pares_y:

        cv2.circle(
            img=imagen,
            center=(int(x), int(y)),
            radius=8,
            color=(255, 50, 50),
            thickness=-1,
        )

    plt.imshow(imagen)
    plt.show()
    return imagen


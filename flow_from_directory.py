import os

import cv2 
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from typing import Generator, Iterator, Tuple

DATA_DIR = Path("data")

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

# def parse_array(str_array: np.ndarray, parser: str = "float") -> np.ndarray:
#     """Parse a string array into a numpy array.

#     Args:
#         str_array (np.ndarray): string array to parse
#         parser (str, optional): parser function to use. Defaults to "float".

#     Returns:
#         np.ndarray: parsed numpy array
#     """    
#     return np.array(list(map(parser, str_array)))



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


def pose_landmark(pose_landmark: dict) -> np.ndarray:
    """Get pose landmark from a dictionary.

    Args:
        pose_landmark (dict): pose landmark dictionary

    Returns:
        np.ndarray: pose landmark as a numpy array
    """

    left_shoulder = pose_landmark["left_shoulder"] # tuple (x, y)

    return np.array([pose_landmark["x"], pose_landmark["y"]])




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



# def flow_from_directory(img_dir: str, labels_dir: str, size: Tuple[int]) -> Generator:

#     with open("all_labels.xml") as f:
#         labels = parse_xml(f)

#     for img_path, labels_ in zip(img_dir, labels):
#         yield Image.open(img_path), labels_




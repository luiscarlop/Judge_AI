import os

import numpy as np
import xml.etree.ElementTree as ET

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

def parse_array(str_array: np.ndarray, parser: str = "float") -> np.ndarray:
    """Parse a string array into a numpy array.

    Args:
        str_array (np.ndarray): string array to parse
        parser (str, optional): parser function to use. Defaults to "float".

    Returns:
        np.ndarray: parsed numpy array
    """    
    return np.array(list(map(parser, str_array)))

def reshape_annotations(annotations: dict) -> Iterator:
    """Flatten annotations into a list of tuples.

    Args:
        annotations (dict): dictionary of annotations

    Returns:
        Iterator: iterator of flattened annotations
    """
    for img_name, labels in annotations.items():
        labels = np.array(labels).flatten().reshape(-1, 2)
        label_list = [parse_array(label) for label in labels]

        annotations[img_name] = label_list
    
    return annotations
    

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
        skeletons = []
        
        for skeleton in image.findall("skeleton"):
            skeleton_nodes = []
            for point in skeleton.findall("points"):
                x = point.get('points').split(',')[0]
                y = point.get('points').split(',')[1]
                skeleton_nodes.append((x, y))
            skeletons.append(skeleton_nodes)
        annotations_dict[img_name] = skeletons
                

    return annotations_dict

def flow_from_directory(img_dir: str, labels_dir: str, size: Tuple[int]) -> Generator:

    with open("all_labels.xml") as f:
        labels = parse_xml(f)

    for img_path, labels_ in zip(img_dir, labels):
        yield Image.open(img_path), labels_


xml_files = [f for f in DATA_DIR.glob("annotations/*.xml")]
combined_root = ET.Element("annotations")

for file in xml_files:
    add_annotations_from_file(file, combined_root)

with open("all_labels.xml", "wb") as f:
    f.write(ET.tostring(combined_root))


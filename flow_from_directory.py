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




import torch
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

def run_detection(image_path: str) -> str:
    # Load model
    model = YOLO(r"ChangeMyclothes\models\FashionYololast.pt")
    
    # Load and process image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run detection
    results = model(image_rgb, conf=0.6)
    
    # Extract bounding boxes
    bboxes = []
    for result in results:
        for box in result.boxes.xyxy.cpu().numpy():
            bboxes.append(box)
            
    # Save bounding boxes
    bbox_path = str(Path(image_path).parent / "bboxes.npy")
    np.save(bbox_path, np.array(bboxes))
    
    return bbox_path

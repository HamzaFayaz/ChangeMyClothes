
# segmentation.py

import torch
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
from pathlib import Path
import cv2

def run_segmentation(image_path: str, bbox_path: str) -> str:
    # Load SAM model
    sam = sam_model_registry["vit_b"](checkpoint=r"ChangeMyclothes\models\sam_vit_b_01ec64.pth")
    sam = sam.to("cuda" if torch.cuda.is_available() else "cpu")
    predictor = SamPredictor(sam)
    
    # Load image and boxes
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    bboxes = np.load(bbox_path)
    
    # Generate mask
    predictor.set_image(image)
    bw_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        input_box = np.array([x1, y1, x2, y2])
        mask, _, _ = predictor.predict(box=input_box, multimask_output=False)
        bw_mask[mask[0]] = 255
    
    # Save mask
    mask_path = str(Path(image_path).parent / "mask.png")
    cv2.imwrite(mask_path, bw_mask)
    
    return mask_path



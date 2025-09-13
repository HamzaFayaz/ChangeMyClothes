from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os
import logging
import shutil
import time
from pathlib import Path
import utils.detection as detection
import utils.segmentation as segmentation
import utils.generation as generation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Setup file paths
BASE_DIR = Path("ChangeMyclothes")
UPLOADS_DIR = BASE_DIR / "uploads"
MASKS_DIR = BASE_DIR / "Mask"
GENERATE_DIR = BASE_DIR / "Generate"

# Ensure directories exist
UPLOADS_DIR.mkdir(exist_ok=True, parents=True)
MASKS_DIR.mkdir(exist_ok=True, parents=True)
GENERATE_DIR.mkdir(exist_ok=True, parents=True)

# Mount static files for image access
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
app.mount("/masks", StaticFiles(directory=str(MASKS_DIR)), name="masks")
app.mount("/generate", StaticFiles(directory=str(GENERATE_DIR)), name="generate")

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Create timestamp for unique filenames
        timestamp = int(time.time())
        
        # Save uploaded file
        file_path = UPLOADS_DIR / f"input_{timestamp}.png"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Image uploaded to: {file_path}")
        
        # Run detection
        detection_result = detection.run_detection(str(file_path))
        bboxes_path = str(UPLOADS_DIR / "bboxes.npy")
        
        # Run segmentation
        mask_path = segmentation.run_segmentation(str(file_path), bboxes_path)
        
        # Create visualization of detection for preview
        detection_image_path = UPLOADS_DIR / f"detection_{timestamp}.png"
        # Implement a function to draw boxes on the image for visualization
        create_detection_preview(str(file_path), bboxes_path, str(detection_image_path))
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "image_path": f"/uploads/{file_path.name}",
                "detection_path": f"/uploads/{detection_image_path.name}",
                "mask_path": f"/uploads/{Path(mask_path).name}"
            }
        )
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/generate")
async def generate_image(
    image_path: str = Form(...),
    mask_path: str = Form(...),
    prompt: str = Form(...),
    negative_prompt: str = Form("")
):
    try:
        # Extract just the filename from the paths
        image_filename = image_path.split('/')[-1]
        mask_filename = mask_path.split('/')[-1]
        
        # Convert to absolute paths
        image_full_path = UPLOADS_DIR / image_filename
        mask_full_path = UPLOADS_DIR / mask_filename  # This should be the segmentation mask
        
        logger.info(f"Generating image with prompt: {prompt}")
        logger.info(f"Using image: {image_full_path}")
        logger.info(f"Using mask: {mask_full_path}")
        
        # Ensure files exist
        if not image_full_path.exists():
            raise HTTPException(status_code=404, detail=f"Image file not found: {image_full_path}")
        if not mask_full_path.exists():
            raise HTTPException(status_code=404, detail=f"Mask file not found: {mask_full_path}")
        
        # Run generation with the mask
        result_path = generation.run_generation(
            str(image_full_path),
            str(mask_full_path),
            prompt,
            negative_prompt
        )
        
        logger.info(f"Generation completed, result saved to: {result_path}")
        
        # Format the URL for frontend
        result_url = f"/generate/{Path(result_path).name}"
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "result_path": result_url
            }
        )
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
    

def create_detection_preview(image_path, bboxes_path, output_path):
    """Create a preview image with detection boxes drawn on it."""
    import cv2
    import numpy as np
    
    # Load image and boxes
    image = cv2.imread(image_path)
    bboxes = np.load(bboxes_path)
    
    # Draw boxes
    for box in bboxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, "main-top", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Save image
    cv2.imwrite(output_path, image)
    return output_path

@app.get("/")
async def root():
    return {"message": "Fashion Image Editor API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

    
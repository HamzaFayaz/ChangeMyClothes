import os
import requests
from pathlib import Path
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv("utils/.env")  # Adjust path as needed

def run_generation(
    image_path: str,
    mask_path: str,
    prompt: str,
    negative_prompt: str = "",
    seed: int = 0
) -> str:
    # API configuration
    stability_key = os.getenv("STABILITY_KEY", "sk-94NnbIhgT4bxci2XrQyON3MNK8G7bOZ8cVoUIaV5t2ffgGR7")
    if not stability_key:
        raise ValueError("STABILITY_KEY environment variable not set")
    
    host = "https://api.stability.ai/v2beta/stable-image/edit/inpaint"
    
    # Ensure mask path is correct - this should be the segmentation mask
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    print(f"Using image: {image_path}")
    print(f"Using mask: {mask_path}")
    
    # Prepare request
    headers = {
        "Accept": "image/*",
        "Authorization": f"Bearer {stability_key}"
    }
    
    files = {
        "image": open(image_path, 'rb'),
        "mask": open(mask_path, 'rb')
    }
    
    params = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "mode": "mask",
        "output_format": "png"
    }
    
    # Send request
    response = requests.post(
        host,
        headers=headers,
        files=files,
        data=params
    )
    
    if not response.ok:
        raise Exception(f"HTTP {response.status_code}: {response.text}")
    
    # Save result to the Generate directory
    output_path = str(Path("ChangeMyclothes/Generate") / f"result_{int(time.time())}.png")
    
    # Make sure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "wb") as f:
        f.write(response.content)
    
    return output_path
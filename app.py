import streamlit as st
import requests
import os
from PIL import Image
import io
import time
import numpy as np

# Define API endpoints
API_URL = "http://localhost:8000"  # Change this if your API is hosted elsewhere

def overlay_images(base_image, overlay_image):
    """Overlay the segmentation mask (in light green) onto the original image for visualization."""
    base = np.array(base_image.convert('RGBA'))
    overlay = np.array(overlay_image.convert('L'))  # Convert mask to grayscale
    
    # Create an RGBA array for the overlay with light green color (RGB: 144, 238, 144)
    overlay_rgba = np.zeros((*overlay.shape, 4), dtype=np.uint8)
    overlay_rgba[:, :, 0] = 144  # Red channel
    overlay_rgba[:, :, 1] = 238  # Green channel
    overlay_rgba[:, :, 2] = 144  # Blue channel
    overlay_rgba[:, :, 3] = (overlay > 0) * 255  # Alpha channel for mask visibility
    
    result = base.copy()
    result[overlay_rgba[:, :, 3] > 0] = overlay_rgba[overlay_rgba[:, :, 3] > 0]  # Apply overlay where mask exists
    return Image.fromarray(result)

def main():
    st.set_page_config(page_title="Fashion Image Editor", layout="wide")
    
    st.title("Change My Clothes - Fashion Image Editor")
    
    # Session state initialization
    if 'uploaded_image_path' not in st.session_state:
        st.session_state.uploaded_image_path = None
    if 'detection_image_path' not in st.session_state:
        st.session_state.detection_image_path = None
    if 'mask_path' not in st.session_state:
        st.session_state.mask_path = None
    if 'result_path' not in st.session_state:
        st.session_state.result_path = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'image_processed' not in st.session_state:
        st.session_state.image_processed = False
    if 'generated' not in st.session_state:
        st.session_state.generated = False
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False
    if 'show_debug' not in st.session_state:
        st.session_state.show_debug = False
    
    # Create main columns for input (left) and output (right)
    input_col, output_col = st.columns([1, 1])
    
    with input_col:
        # Input prompt field above upload field
        st.header("Describe Your New Clothing")
        prompt = st.text_input("What do you want the clothing to look like?", 
                           value="",
                           max_chars=50)  # Reduced length of input field
        
        # Upload section below prompt
        st.header("Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
        
        # Process uploaded image automatically for detection and segmentation (only once)
        if uploaded_file is not None and not st.session_state.processing and not st.session_state.image_processed:
            st.session_state.processing = True
            
            with st.spinner("Processing image..."):
                # Upload image to API for detection and segmentation
                files = {"file": uploaded_file.getvalue()}
                response = requests.post(f"{API_URL}/upload", 
                                      files={"file": ("image.png", uploaded_file.getvalue(), "image/png")})
                
                if response.status_code == 200:
                    result = response.json()
                    st.session_state.uploaded_image_path = result["image_path"]
                    st.session_state.detection_image_path = result["detection_path"]
                    st.session_state.mask_path = result["mask_path"]
                    st.session_state.image_processed = True
                    st.success("Image processed successfully!")
                else:
                    st.error(f"Error processing image: {response.text}")
                
                st.session_state.processing = False
        
        # Show Generate button only after image is processed
        if st.session_state.image_processed:
            generate_button = st.button("Generate New Image", 
                                     disabled=False,
                                     type="primary")
            
            # Handle generation (only generative model, no reprocessing)
            if generate_button and not st.session_state.generated:
                st.session_state.generated = True
                
                with st.spinner("Generating new image..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/generate",
                            data={
                                "image_path": st.session_state.uploaded_image_path,
                                "mask_path": st.session_state.mask_path,
                                "prompt": prompt,
                                "negative_prompt": ""  # Empty negative prompt
                            }
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.session_state.result_path = result["result_path"]
                            st.success("Image generated successfully!")
                            st.session_state.show_results = True
                        else:
                            st.error(f"Error generating image: {response.text}")
                    except Exception as e:
                        st.error(f"Error during generation: {str(e)}")
                    
                    st.session_state.generated = False
            
            # Add Debug/Visual button below Generate New Image
            if st.button("Debug/Visual"):
                st.session_state.show_debug = True
    
    # Display results in a professional size on the right side after generation is complete
    with output_col:
        if st.session_state.show_results:
            st.header("Results")
            col1, col2 = st.columns(2)  # Two columns for original and generated images
            
            with col1:
                st.subheader("Original Image")
                try:
                    st.image(uploaded_file, use_container_width=True)  # Professional, fixed width
                except Exception as e:
                    st.error(f"Error displaying original image: {str(e)}")
            
            with col2:
                st.subheader("Generated Result")
                try:
                    if st.session_state.result_path:
                        result_url = f"{API_URL}{st.session_state.result_path}"
                        result_response = requests.get(result_url)
                        if result_response.status_code == 200:
                            result_image = Image.open(io.BytesIO(result_response.content))
                            st.image(result_image, use_container_width=True)  # Professional, fixed width
                            
                            # Download button for the generated image
                            result_img_bytes = io.BytesIO(result_response.content)
                            st.download_button(
                                label="Download Generated Image",
                                data=result_img_bytes,
                                file_name="generated_clothing.png",
                                mime="image/png"
                            )
                except Exception as e:
                    st.error(f"Error displaying result: {str(e)}")
    
    # Display debug/visual images in a row below the Debug/Visual button, spanning the full width
    if st.session_state.show_debug and st.session_state.image_processed:
        st.header("Debug/Visual Results")
        debug_col1, debug_col2, debug_col3, debug_col4 = st.columns(4)  # Four columns spanning the full row
        
        with debug_col1:
            st.subheader("Original Image")
            try:
                st.image(uploaded_file, use_container_width=True)  # Full container width for professional look
            except Exception as e:
                st.error(f"Error displaying original image: {str(e)}")
        
        with debug_col2:
            st.subheader("Detection Result")
            try:
                if st.session_state.detection_image_path:
                    detection_url = f"{API_URL}{st.session_state.detection_image_path}"
                    detection_response = requests.get(detection_url)
                    if detection_response.status_code == 200:
                        detection_image = Image.open(io.BytesIO(detection_response.content))
                        st.image(detection_image, use_container_width=True)  # Full container width
            except Exception as e:
                st.error(f"Error displaying detection: {str(e)}")
        
        with debug_col3:
            st.subheader("Segmentation Overlay")
            try:
                if st.session_state.mask_path and uploaded_file:
                    mask_url = f"{API_URL}{st.session_state.mask_path}"
                    mask_response = requests.get(mask_url)
                    if mask_response.status_code == 200:
                        mask_image = Image.open(io.BytesIO(mask_response.content))
                        original_image = Image.open(uploaded_file)
                        overlay_image = overlay_images(original_image, mask_image)
                        st.image(overlay_image, use_container_width=True)  # Full container width
            except Exception as e:
                st.error(f"Error displaying segmentation overlay: {str(e)}")
        
        with debug_col4:
            st.subheader("Generated Result")
            try:
                if st.session_state.result_path:
                    result_url = f"{API_URL}{st.session_state.result_path}"
                    result_response = requests.get(result_url)
                    if result_response.status_code == 200:
                        result_image = Image.open(io.BytesIO(result_response.content))
                        st.image(result_image, use_container_width=True)  # Full container width
            except Exception as e:
                st.error(f"Error displaying result: {str(e)}")

if __name__ == "__main__":
    main()
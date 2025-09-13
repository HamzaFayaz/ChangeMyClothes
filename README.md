# Change My Clothes - AI Fashion Editor
YOLO SAM Stability AI FastAPI Streamlit OpenCV

## 📋 About This Project
An advanced AI-powered fashion image editing system built with modular architecture for real-time clothing detection, precise segmentation, and intelligent inpainting. The project leverages YOLO for high-performance clothing detection, SAM (Segment Anything Model) for pixel-perfect segmentation, and Stability AI for professional-quality clothing generation with a modern web interface.

## What Makes It Special:
🎯 **Real-Time Detection**: Ultra-fast YOLO object detection with custom-trained fashion model for accurate clothing identification.
🎨 **Precise Segmentation**: Meta's SAM model provides pixel-perfect clothing masks for seamless editing.
🖼️ **AI-Powered Generation**: Stability AI inpainting creates realistic new clothing based on text prompts.
🌐 **Modern Web Interface**: Professional Streamlit frontend with real-time processing and debug visualization.
⚡ **High Performance**: Optimized pipeline with CUDA acceleration and efficient image processing.
🎛️ **Configurable Settings**: Adjustable detection confidence, prompt customization, and negative prompts.
🔧 **Modular Architecture**: Clean separation between detection, segmentation, generation, and web components.
🎨 **Professional Results**: High-quality clothing replacement with natural lighting and texture preservation.
📊 **Live Monitoring**: Real-time processing status, debug visualization, and result comparison.
🛡️ **Robust Design**: Comprehensive error handling and file management for production stability.
👗 **Fashion-Focused**: Specialized for clothing items with intelligent detection and generation algorithms.

## 🎬 Demo Video

Watch our comprehensive demo showcasing the complete Change My Clothes workflow:

[![Change My Clothes Demo](https://img.youtube.com/vi/YOUR_VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)

**What you'll see in the demo:**
- 🖼️ **Image Upload**: Seamless fashion photo upload through the web interface
- 🎯 **Real-Time Detection**: YOLO automatically identifying clothing items with bounding boxes
- 🎨 **Precise Segmentation**: SAM creating pixel-perfect masks for clothing regions
- ✍️ **Text Prompting**: Entering creative descriptions for desired clothing changes
- 🖼️ **AI Generation**: Stability AI creating realistic new clothing based on prompts
- 🔍 **Debug Visualization**: Step-by-step inspection of the entire pipeline
- 📥 **Result Download**: Saving the final generated image

## 🚀 Key Features

### Core Functionality

🧠 **AI-Powered Detection**: YOLO neural network for real-time clothing detection with CUDA acceleration.
🎨 **Precise Segmentation**: SAM (Segment Anything Model) for pixel-perfect clothing masks and boundaries.
🖼️ **Intelligent Generation**: Stability AI inpainting for realistic clothing replacement based on text prompts.
🪄 **Modular Architecture**: Clean separation between detection logic, segmentation, generation, and web components.
⚡ **High-Performance Processing**: Optimized pipeline with GPU acceleration and efficient image processing.
📊 **Real-time Monitoring**: Live processing status, debug visualization, and result comparison.
🔌 **Seamless Integration**: Built-in support for FastAPI backend and Streamlit frontend communication.

### Web Interface Features

🎨 **Professional Interface**: Modern Streamlit GUI with clean, intuitive design and responsive layout.
📱 **Interactive Controls**: Real-time processing buttons and status indicators with visual feedback.
📂 **Live Statistics**: Real-time processing status and debug information display.
💬 **Status Updates**: Live detection status and segmentation progress information.
🔄 **Dynamic Controls**: Toggle buttons for debug mode and result visualization.
🧩 **Settings Panel**: Configurable detection confidence, prompt customization, and negative prompts.

### AI Pipeline Features

⚡ **YOLO Detection Engine**: Custom-trained fashion detection model with CUDA support.
🎨 **SAM Integration**: Meta's Segment Anything Model for precise clothing segmentation.
🖼️ **Stability AI Generation**: Professional-quality inpainting for realistic clothing replacement.
🧠 **Smart Processing**: Intelligent pipeline with automatic detection and segmentation workflow.
🔄 **File Management**: Optimized file handling for uploads, masks, and generated results.
📜 **Error Handling**: Robust error management and user feedback mechanisms.
🐳 **Production Ready**: Professional deployment structure with comprehensive logging and monitoring.

---

## 📁 Project Structure

```
ChangeMyclothes/
├── main.py                 # FastAPI backend server
├── app.py                  # Streamlit frontend interface
├── models/                 # AI model files
│   ├── FashionYololast.pt      # Custom YOLO fashion detection model
│   ├── sam_vit_b_01ec64.pth    # SAM ViT-B segmentation model
│   ├── sam_vit_h.pth           # SAM ViT-H segmentation model
│   └── sam_vit_l_0b3195.pth    # SAM ViT-L segmentation model
├── utils/                  # Core AI pipeline modules
│   ├── detection.py            # YOLO detection implementation
│   ├── segmentation.py         # SAM segmentation implementation
│   └── generation.py           # Stability AI inpainting implementation
├── uploads/                # User uploaded images
├── Mask/                   # Generated segmentation masks
├── Generate/               # Final generated images
├── Experiment_images/      # Test images and experimental results
└── README.md              # Project documentation
```

## 🔧 API Endpoints

### FastAPI Backend (`main.py`)

- **POST `/upload`**: Upload image and run detection + segmentation
  - Input: Image file
  - Output: Image path, detection preview, mask path

- **POST `/generate`**: Generate new clothing using inpainting
  - Input: Image path, mask path, prompt, negative prompt
  - Output: Generated image path

- **GET `/`**: Health check endpoint

## 🎯 Technical Details

### AI Pipeline Architecture

1. **Detection Stage**: YOLO identifies clothing items with bounding boxes
2. **Segmentation Stage**: SAM creates precise pixel-level masks
3. **Generation Stage**: Stability AI inpaints new clothing based on masks and prompts

### Model Specifications

- **YOLO Model**: Custom-trained for fashion detection (confidence threshold: 0.6)
- **SAM Model**: ViT-B backbone with CUDA acceleration
- **Stability AI**: Professional inpainting API with mask mode

### Performance Optimizations

- GPU acceleration for YOLO and SAM models
- Efficient file handling and caching
- Asynchronous API processing
- Optimized image preprocessing

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce image size or use CPU mode
2. **API Key Error**: Ensure Stability AI key is properly set in `.env`
3. **Model Loading Error**: Verify model files are in the correct directory
4. **File Not Found**: Check that upload directories exist and have proper permissions

### Debug Mode

Use the "Debug/Visual" button in the interface to inspect:
- Original image
- Detection results with bounding boxes
- Segmentation mask overlay
- Final generated result

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended)
- Git (for cloning the repository)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/ChangeMyclothes.git
cd ChangeMyclothes
```

### Step 2: Download YOLO Model
Download the custom-trained YOLO fashion detection model:
- **Model**: `FashionYololast.pt`
- **Size**: ~50MB
- **Purpose**: Clothing detection and bounding box generation
- **Place in**: `models/FashionYololast.pt`

### Step 3: Download SAM Models
Download the Segment Anything Model checkpoints:

**SAM ViT-B (Recommended)**:
- **Model**: `sam_vit_b_01ec64.pth`
- **Size**: ~375MB
- **Purpose**: Primary segmentation model

**SAM ViT-H (High Quality)**:
- **Model**: `sam_vit_h.pth`
- **Size**: ~2.4GB
- **Purpose**: Higher quality segmentation (optional)

**SAM ViT-L (Large)**:
- **Model**: `sam_vit_l_0b3195.pth`
- **Size**: ~1.2GB
- **Purpose**: Large model for better accuracy (optional)

**Download from**: [Meta SAM GitHub Releases](https://github.com/facebookresearch/segment-anything/releases)

**Place all models in**: `models/` directory

### Step 4: Get Stability AI API Key
1. Visit [Stability AI Platform](https://platform.stability.ai/account/keys)
2. Sign up or log in to your account
3. Navigate to the API Keys section
4. Generate a new API key
5. Copy the key for the next step

### Step 5: Set Up Environment Variables
Create a `.env` file in the `root/` directory:
```bash
# Create .env file in utils/ directory
echo "STABILITY_KEY=your_stability_ai_api_key_here" > root/.env
```

Replace `your_stability_ai_api_key_here` with your actual API key from Step 4.

### Step 6: Install Python Dependencies
```bash
pip install -r requirements.txt
```

**Required packages**:
- `fastapi` - Web framework for the API
- `streamlit` - Frontend web interface
- `ultralytics` - YOLO model inference
- `segment-anything` - SAM model integration
- `opencv-python` - Image processing
- `numpy` - Numerical operations
- `pillow` - Image manipulation
- `requests` - HTTP requests for Stability AI
- `python-dotenv` - Environment variable management
- `uvicorn` - ASGI server for FastAPI

### Step 7: Verify Installation
Check that all models are in place:
```bash
ls models/
# Should show:
# FashionYololast.pt
# sam_vit_b_01ec64.pth
# sam_vit_h.pth (optional)
# sam_vit_l_0b3195.pth (optional)
```

### Step 8: Test the Installation
1. **Start the FastAPI backend**:
```bash
python main.py
```

2. **In a new terminal, start the Streamlit frontend**:
```bash
streamlit run app.py
```

3. **Open your browser** and navigate to `http://localhost:8501`

### Step 9: First Run Setup
- Upload a test image through the web interface
- Verify that detection and segmentation work
- Test the generation feature with a simple prompt
- Check the debug mode to see all pipeline stages

### Troubleshooting Installation
- **CUDA Issues**: Ensure you have the correct CUDA version installed
- **Model Loading Errors**: Verify all model files are in the `models/` directory
- **API Key Issues**: Double-check your Stability AI key in the `.env` file
- **Permission Errors**: Ensure the application has write permissions for uploads and generated files



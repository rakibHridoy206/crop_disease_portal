from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os, shutil, cv2
from hf_model import CropDiseaseViT
import numpy as np
from skimage.metrics import structural_similarity as ssim

app = FastAPI()

# Static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

model_handler = CropDiseaseViT()

def is_similar(img1, img2, threshold=0.95):
    """Compare two images for similarity using SSIM"""
    try:
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Resize images to same size if needed
        if gray1.shape != gray2.shape:
            gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
            
        # Calculate SSIM
        score = ssim(gray1, gray2)
        return score > threshold
    except Exception as e:
        print(f"Error comparing images: {e}")
        return False

def extract_frames(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved_frames = []
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    previous_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % fps == 0:  # 1 frame per second
            # Check if frame is similar to previous frame
            is_duplicate = False
            if previous_frame is not None:
                is_duplicate = is_similar(previous_frame, frame)
                
            if not is_duplicate:
                frame_name = os.path.join(output_folder, f"frame_{count}.jpg")
                cv2.imwrite(frame_name, frame)
                saved_frames.append(frame_name)
                previous_frame = frame.copy()
                
        count += 1
    cap.release()
    return saved_frames

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "message": None,
        "video_path": None,
        "results": []
    })

@app.post("/upload", response_class=HTMLResponse)
async def upload_video(request: Request, file: UploadFile = File(...)):
    upload_dir = "static/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    video_path = os.path.join(upload_dir, file.filename)

    # Save uploaded video
    with open(video_path, "wb") as f:
        f.write(await file.read())

    # Extract frames
    frames_dir = "static/frames"
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir)

    frame_files = extract_frames(video_path, frames_dir)

    # Predict
    results = []
    for frame in frame_files:
        pred = model_handler.predict(frame)
        results.append({
            "frame": "/" + frame.replace("\\", "/"),
            "label": pred["label"]
        })

    return templates.TemplateResponse("index.html", {
        "request": request,
        "message": "Video uploaded & analyzed!",
        "video_path": "/" + video_path.replace("\\", "/"),
        "results": results
    })

@app.post("/clear", response_class=HTMLResponse)
async def clear_data(request: Request):
    # Delete files in uploads and frames folders
    for folder in ["static/uploads", "static/frames"]:
        if os.path.exists(folder):
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "message": "All uploaded videos and frames have been cleared.",
        "video_path": None,
        "results": []
    })


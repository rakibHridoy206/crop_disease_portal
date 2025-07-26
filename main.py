from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os, shutil, cv2
from hf_model import CropDiseaseViT

app = FastAPI()

# Static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

model_handler = CropDiseaseViT()

def extract_frames(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved_frames = []
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % fps == 0:  # 1 frame per second
            frame_name = os.path.join(output_folder, f"frame_{count}.jpg")
            cv2.imwrite(frame_name, frame)
            saved_frames.append(frame_name)
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


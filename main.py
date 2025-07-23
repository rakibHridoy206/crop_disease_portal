from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import os
import cv2
from model_predict import predict_frame_real

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "static/uploads"
FRAME_DIR = "static/frames"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FRAME_DIR, exist_ok=True)


def extract_frames(video_path: str, frame_dir: str, interval: int = 1):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)

    count = 0
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        if count % frame_interval == 0:
            frame_filename = f"frame_{frame_count}.jpg"
            frame_path = os.path.join(frame_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            frame_count += 1
        count += 1
    cap.release()
    return frame_count


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload", response_class=HTMLResponse)
async def upload_video(request: Request, file: UploadFile = File(...)):
    video_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Clear old frames
    for f in os.listdir(FRAME_DIR):
        os.remove(os.path.join(FRAME_DIR, f))

    # Extract frames
    frame_count = extract_frames(video_path, FRAME_DIR)

    frame_urls = [f"/static/frames/frame_{i}.jpg" for i in range(frame_count)]

    results = []


    for frame_url in frame_urls:
        frame_path = frame_url.replace("/static/", "static/")
        label = predict_frame_real(frame_path)
        results.append({"frame": frame_url, "label": label})

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "video_path": f"/static/uploads/{file.filename}",
            "message": f"Video uploaded and {frame_count} frames analyzed.",
            "results": results,
        },
    )

# return templates.TemplateResponse("index.html", {
#     "request": request,
#     "video_path": f"/static/uploads/{file.filename}",
#     "message": f"Video uploaded and {frame_count} frames extracted.",
#     "frames": frame_urls
# })

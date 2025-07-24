from fastapi import FastAPI, UploadFile, File
from hf_model import load_model, predict_image
from PIL import Image
import os
import cv2
from collections import Counter
import shutil

app = FastAPI()

# Load the model only once at startup
model, transform, labels = load_model()


def extract_frames(video_path, output_folder, frame_rate=1):
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = 0

    os.makedirs(output_folder, exist_ok=True)

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Save one frame every second
        if count % int(cap.get(cv2.CAP_PROP_FPS)) == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved += 1

        count += 1

    cap.release()


@app.post("/analyze-video/")
async def analyze_video(file: UploadFile = File(...)):
    video_path = f"temp_videos/{file.filename}"
    os.makedirs("temp_videos", exist_ok=True)

    with open(video_path, "wb") as f:
        f.write(await file.read())

    os.makedirs("extracted_frames", exist_ok=True)
    extract_frames(video_path, "extracted_frames", frame_rate=1)

    predictions = []
    for frame_file in sorted(os.listdir("extracted_frames")):
        frame_path = os.path.join("extracted_frames", frame_file)
        try:
            image = Image.open(frame_path).convert("RGB")
            prediction = predict_image(model, transform, labels, image)
            predictions.append(prediction)
        except Exception as e:
            predictions.append(f"Error: {str(e)}")

    most_common = Counter(predictions).most_common(1)[0]

    # Cleanup
    os.remove(video_path)
    shutil.rmtree("extracted_frames")

    return {
        "total_frames": len(predictions),
        "most_common_disease": most_common[0],
        "count": most_common[1],
        "all_predictions": predictions
    }
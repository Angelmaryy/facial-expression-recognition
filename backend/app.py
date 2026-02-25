import os
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# -----------------------------------------------
# LOAD MODEL
# -----------------------------------------------
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "emotion_model.h5")

print(f"Loading model from: {MODEL_PATH}")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# Emotion labels (FER2013 order)
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -----------------------------------------------
# FASTAPI APP
# -----------------------------------------------
app = FastAPI(title="Facial Emotion Recognition API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Change to your frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------
# HELPERS
# -----------------------------------------------

def detect_and_predict(image_bgr: np.ndarray):
    """
    Given a BGR image (numpy array), detect faces and predict emotions.
    Returns annotated image (BGR) + list of face result dicts.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    results = []

    for (x, y, w, h) in faces_rect:
        face_roi = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (48, 48))
        face_normalized = face_resized / 255.0
        face_input = np.reshape(face_normalized, (1, 48, 48, 1))

        preds = model.predict(face_input, verbose=0)[0]
        emotion_idx = int(np.argmax(preds))
        emotion = EMOTION_LABELS[emotion_idx]
        confidence = float(preds[emotion_idx])

        all_scores = {EMOTION_LABELS[i]: float(preds[i]) for i in range(7)}

        results.append({
            "box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "emotion": emotion,
            "confidence": round(confidence * 100, 2),
            "scores": {k: round(v * 100, 2) for k, v in all_scores.items()}
        })

        # Annotate image
        color = (0, 255, 128)
        cv2.rectangle(image_bgr, (x, y), (x+w, y+h), color, 2)
        label = f"{emotion} {confidence*100:.1f}%"
        cv2.putText(image_bgr, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    return image_bgr, results


def encode_image_to_base64(image_bgr: np.ndarray) -> str:
    _, buffer = cv2.imencode(".jpg", image_bgr)
    return base64.b64encode(buffer).decode("utf-8")


# -----------------------------------------------
# ROUTES
# -----------------------------------------------

@app.get("/")
def root():
    return {"message": "Facial Emotion Recognition API is running 🚀"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    """
    Upload an image file → get emotion predictions + annotated image.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image file"})

    annotated, results = detect_and_predict(image)
    annotated_b64 = encode_image_to_base64(annotated)

    return {
        "faces_detected": len(results),
        "results": results,
        "annotated_image": f"data:image/jpeg;base64,{annotated_b64}"
    }


@app.websocket("/ws/webcam")
async def websocket_webcam(websocket: WebSocket):
    """
    WebSocket endpoint for real-time webcam emotion detection.
    Client sends base64-encoded JPEG frames → server replies with predictions + annotated frame.
    """
    await websocket.accept()
    print("📡 WebSocket client connected")

    try:
        while True:
            # Receive base64 frame from client
            data = await websocket.receive_text()

            # Decode base64 → numpy image
            if "," in data:
                data = data.split(",")[1]   # strip data:image/... prefix
            img_bytes = base64.b64decode(data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                await websocket.send_json({"error": "Bad frame"})
                continue

            annotated, results = detect_and_predict(image)
            annotated_b64 = encode_image_to_base64(annotated)

            await websocket.send_json({
                "faces_detected": len(results),
                "results": results,
                "annotated_image": f"data:image/jpeg;base64,{annotated_b64}"
            })

    except WebSocketDisconnect:
        print("📡 WebSocket client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()


# -----------------------------------------------
# RUN
# -----------------------------------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)

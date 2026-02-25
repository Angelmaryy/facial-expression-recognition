# 🧠 EmotiScan — Facial Emotion Recognition System

An end-to-end **AI-powered Facial Emotion Recognition** web application that detects human emotions from images and live webcam streams using Deep Learning and Computer Vision.

The system uses a CNN model trained on the **FER2013 dataset** and provides real-time emotion analysis through a modern web interface powered by **FastAPI + WebSockets**.

---

## 🚀 Features

✅ Real-time emotion detection using webcam
✅ Image upload emotion analysis
✅ Deep Learning CNN model (TensorFlow/Keras)
✅ Face detection using OpenCV Haar Cascades
✅ Live WebSocket streaming for low-latency prediction
✅ Annotated output with confidence scores
✅ Interactive frontend dashboard (EmotiScan UI)

---

## 🧠 Supported Emotions

The model predicts the following emotions:

* Angry 😠
* Disgust 🤢
* Fear 😨
* Happy 😀
* Sad 😢
* Surprise 😲
* Neutral 😐

---

## 🏗️ System Architecture

```
Frontend (HTML + JavaScript)
        │
        ├── REST API → Image Upload Prediction
        │
        └── WebSocket → Live Webcam Frames
                     │
                FastAPI Backend
                     │
              CNN Emotion Model
                     │
                FER2013 Dataset
```

---

## 📂 Project Structure

```
facial-expression-recognition
│
├── backend/
│   └── app.py              # FastAPI backend + WebSocket server
│
├── frontend/
│   └── index.html          # EmotiScan web interface
│
├── model/
│   └── emotion_model.h5    # Trained CNN model
│
├── src/
│   ├── train.py            # Model training script
│   ├── realtime.py         # OpenCV realtime detection
│   └── upload_detect.py    # Image detection script
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Angelmarry/facial-expression-recognition.git
cd facial-expression-recognition
```

---

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / Mac**

```bash
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

### ✅ Start Backend (FastAPI)

```bash
cd backend
python app.py
```

Server runs at:

```
http://localhost:8000
```

---

### ✅ Start Frontend

From project root:

```bash
python -m http.server 5500
```

Open browser:

```
http://localhost:5500/frontend/index.html
```

---

## 📷 Usage

### Live Webcam Detection

1. Open **Live Webcam** tab
2. Click **Start Detection**
3. Allow camera permission
4. Real-time emotion predictions appear

### Image Upload Detection

1. Switch to **Upload Image**
2. Select an image containing a face
3. Click **Analyze Emotion**

---

## 🧩 Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* FastAPI
* WebSockets
* HTML5 / JavaScript
* FER2013 Dataset

---

## 📊 Model Details

* Input Size: **48 × 48 grayscale**
* Architecture: Convolutional Neural Network (CNN)
* Dataset: FER2013
* Training Accuracy: ~70%
* Real-time inference supported

---

## 🔮 Future Improvements

* Emotion tracking timeline graph
* Face tracking stabilization
* Mobile responsive UI
* Cloud deployment (Render / Railway)
* Transformer-based emotion models

---

## 👩‍💻 Author

**Angel Mary**

B.Tech Electrical and Electronics Engineering

---

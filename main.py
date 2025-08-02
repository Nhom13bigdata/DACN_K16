from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import cv2
from PIL import Image
import io
import os
import datetime

from face_detection import FaceDetector
from face_recognition import FaceRecognizer
from vector_database import VectorDB

# --- Configuration --- #
# Path for YOLOv8 face detection model
# Download yolov8n-face.pt or similar pre-trained YOLOv8 face model
# and place it in the \'models\' directory.
YOLO_MODEL_PATH = "./models/yolov8n-face.pt"

# MobileFaceNet model path (not directly used as facenet_pytorch handles loading)
# The InceptionResnetV1 model from facenet_pytorch is used, which is pre-trained.
MOBILEFACENET_MODEL_PATH = None 

# Vector Database configuration
VECTOR_DB_DIMENSION = 512 # InceptionResnetV1 outputs 512-dimensional embeddings
VECTOR_DB_PATH = "./data/face_embeddings.bin"
ID_MAP_PATH = "./data/id_map.json"

# Ensure data directory exists
os.makedirs("./data", exist_ok=True)

# --- Initialize Models and Database --- #
face_detector = FaceDetector(model_path=YOLO_MODEL_PATH)
face_recognizer = FaceRecognizer(model_path=MOBILEFACENET_MODEL_PATH) # model_path is ignored by FaceRecognizer
vector_db = VectorDB(dimension=VECTOR_DB_DIMENSION, db_path=VECTOR_DB_PATH, id_map_path=ID_MAP_PATH)

app = FastAPI()

# --- Pydantic Models for Request/Response --- #
class RegisterFaceResponse(BaseModel):
    user_id: str
    user_name: str
    message: str

class CheckInResponse(BaseModel):
    status: str
    recognized_faces: list # List of dicts {user_id, user_name, similarity}
    message: str

class AttendanceRecord(BaseModel):
    user_id: str
    user_name: str
    timestamp: datetime.datetime
    status: str = "checked_in"

class GetAttendanceResponse(BaseModel):
    status: str
    attendance_records: list[AttendanceRecord]

# In-memory attendance records (for demonstration, replace with a proper DB in production)
attendance_log = []

# --- Helper Functions --- #
def read_image_from_bytes(image_bytes: bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return np.array(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

# --- API Endpoints --- #
@app.post("/register", response_model=RegisterFaceResponse)
async def register_face(user_id: str, user_name: str, file: UploadFile = File(...)):
    """
    Đăng ký khuôn mặt mới vào hệ thống.
    - `user_id`: ID duy nhất của người dùng.
    - `user_name`: Tên của người dùng.
    - `file`: Ảnh khuôn mặt của người dùng (JPEG/PNG).
    """
    image_bytes = await file.read()
    image_np = read_image_from_bytes(image_bytes)

    face_boxes = face_detector.detect_faces(image_np)

    if not face_boxes:
        raise HTTPException(status_code=400, detail="Không tìm thấy khuôn mặt nào trong ảnh.")
    if len(face_boxes) > 1:
        # For registration, we typically expect only one face
        raise HTTPException(status_code=400, detail="Tìm thấy nhiều hơn một khuôn mặt. Vui lòng cung cấp ảnh chỉ có một khuôn mặt.")

    # Take the first (and only) detected face
    face_box = face_boxes[0]
    extracted_face = face_detector.extract_face_from_image(image_np, face_box)

    if extracted_face is None or extracted_face.size == 0:
        raise HTTPException(status_code=500, detail="Không thể trích xuất khuôn mặt từ ảnh.")

    embedding = face_recognizer.get_face_embedding(extracted_face)

    if embedding is None:
        raise HTTPException(status_code=500, detail="Không thể tạo embedding cho khuôn mặt.")

    # Add face to vector database
    vector_db.add_face(user_id, user_name, embedding)

    return RegisterFaceResponse(user_id=user_id, user_name=user_name, message="Đăng ký khuôn mặt thành công.")

@app.post("/check_in", response_model=CheckInResponse)
async def check_in(file: UploadFile = File(...)):
    """
    Thực hiện điểm danh bằng cách nhận dạng khuôn mặt từ ảnh hoặc luồng video.
    - `file`: Ảnh chứa khuôn mặt cần điểm danh (JPEG/PNG).
    """
    image_bytes = await file.read()
    image_np = read_image_from_bytes(image_bytes)

    face_boxes = face_detector.detect_faces(image_np)

    recognized_faces = []
    if not face_boxes:
        return CheckInResponse(status="failed", recognized_faces=[], message="Không tìm thấy khuôn mặt nào trong ảnh.")

    for face_box in face_boxes:
        extracted_face = face_detector.extract_face_from_image(image_np, face_box)
        if extracted_face is None or extracted_face.size == 0:
            continue

        embedding = face_recognizer.get_face_embedding(extracted_face)
        if embedding is None:
            continue

        # Search for the most similar face in the vector database
        search_results = vector_db.search_face(embedding, k=1, threshold=0.7) # Adjust threshold as needed for cosine similarity

        if search_results:
            best_match = search_results[0]
            user_id = best_match["user_id"]
            user_name = best_match["user_name"]
            similarity = best_match["similarity"]
            
            # Log attendance
            attendance_log.append(AttendanceRecord(
                user_id=user_id,
                user_name=user_name,
                timestamp=datetime.datetime.now()
            ))
            recognized_faces.append({"user_id": user_id, "user_name": user_name, "similarity": similarity})
        else:
            recognized_faces.append({"user_id": "unknown", "user_name": "Unknown", "similarity": 0.0})

    if recognized_faces:
        return CheckInResponse(status="success", recognized_faces=recognized_faces, message="Điểm danh thành công.")
    else:
        return CheckInResponse(status="failed", recognized_faces=[], message="Không nhận dạng được khuôn mặt nào.")

@app.get("/get_attendance", response_model=GetAttendanceResponse)
async def get_attendance(date: str = None, user_id: str = None):
    """
    Lấy lịch sử điểm danh.
    - `date`: Ngày cụ thể để lọc (YYYY-MM-DD, tùy chọn).
    - `user_id`: ID người dùng để lọc (tùy chọn).
    """
    filtered_records = attendance_log

    if date:
        try:
            target_date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
            filtered_records = [rec for rec in filtered_records if rec.timestamp.date() == target_date]
        except ValueError:
            raise HTTPException(status_code=400, detail="Định dạng ngày không hợp lệ. Vui lòng sử dụng YYYY-MM-DD.")

    if user_id:
        filtered_records = [rec for rec in filtered_records if rec.user_id == user_id]

    return GetAttendanceResponse(status="success", attendance_records=filtered_records)

# To run this API:
# 1. Save this file as main.py in the attendance_system directory.
# 2. Make sure you have face_detection.py, face_recognition.py, and vector_database.py in the same directory.
# 3. Download yolov8n-face.pt (or similar) into the \'models\' directory.
# 4. Install dependencies: pip install -r requirements.txt
# 5. Run: uvicorn main:app --host 0.0.0.0 --port 8000



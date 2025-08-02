# Hệ Thống Điểm Danh Tự Động với YOLOv8, MobileFaceNet và Vector Database

Đây là một hệ thống điểm danh tự động được xây dựng bằng Python, sử dụng các công nghệ thị giác máy tính tiên tiến như YOLOv8 để phát hiện khuôn mặt, MobileFaceNet (thông qua `facenet_pytorch` sử dụng InceptionResnetV1) để nhận dạng khuôn mặt và một Vector Database (FAISS) để lưu trữ và tìm kiếm các đặc trưng khuôn mặt. Hệ thống cung cấp một API RESTful được xây dựng với FastAPI để quản lý các hoạt động điểm danh và đăng ký người dùng.

## Cấu trúc dự án

```
attendance_system/
├── models/                  # Chứa các file model (yolov8n-face.pt)
├── data/                    # Chứa dữ liệu vector database (face_embeddings.bin, id_map.json)
├── face_detection.py        # Module phát hiện khuôn mặt với YOLOv8
├── face_recognition.py      # Module nhận dạng khuôn mặt với MobileFaceNet (InceptionResnetV1)
├── vector_database.py       # Module quản lý Vector Database (FAISS)
├── main.py                  # API chính của hệ thống (FastAPI)
├── requirements.txt         # Danh sách các thư viện Python cần thiết
└── README.md                # File hướng dẫn sử dụng
```

## Cài đặt

Để cài đặt và chạy hệ thống, bạn cần thực hiện các bước sau:

### 1. Clone repository (nếu có)

```bash
git clone <your-repository-url>
cd attendance_system
```

### 2. Cài đặt các thư viện Python

Đảm bảo bạn đã cài đặt `pip` và `python3`.

```bash
pip install -r requirements.txt
```

### 3. Tải xuống các mô hình

#### YOLOv8 cho phát hiện khuôn mặt

Hệ thống này sử dụng mô hình YOLOv8 để phát hiện khuôn mặt. Bạn cần tải xuống một mô hình YOLOv8 đã được huấn luyện chuyên biệt cho việc phát hiện khuôn mặt (ví dụ: `yolov8n-face.pt`). Nếu không có mô hình `yolov8n-face.pt` cụ thể, bạn có thể sử dụng các mô hình YOLOv8 tổng quát (như `yolov8n.pt` từ Ultralytics) và sau đó lọc các đối tượng là người, rồi áp dụng một thuật toán phát hiện khuôn mặt khác trên vùng người được phát hiện. Tuy nhiên, để đơn giản và hiệu quả, khuyến nghị sử dụng mô hình đã được huấn luyện cho khuôn mặt.

Bạn có thể tìm kiếm và tải xuống các mô hình YOLOv8 đã được huấn luyện cho khuôn mặt từ các nguồn như Hugging Face hoặc GitHub của Ultralytics hoặc các dự án cộng đồng khác. Đặt file `.pt` đã tải xuống vào thư mục `models/`.

Ví dụ (thay thế bằng link tải thực tế của mô hình YOLOv8-face):

```bash
# Ví dụ tải file yolov8n-face.pt
# wget -P models/ https://example.com/path/to/yolov8n-face.pt
```

#### MobileFaceNet (InceptionResnetV1)

Module nhận dạng khuôn mặt sử dụng `InceptionResnetV1` từ thư viện `facenet_pytorch`. Mô hình này sẽ được tự động tải xuống khi bạn khởi tạo `FaceRecognizer` nếu nó chưa có sẵn. Do đó, bạn không cần phải tải file `.pth` riêng cho MobileFaceNet.

## Chạy ứng dụng

Sau khi cài đặt và tải mô hình, bạn có thể chạy API server bằng `uvicorn`:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

*   `main`: Tên file Python chứa ứng dụng FastAPI (main.py).
*   `app`: Tên biến ứng dụng FastAPI trong file `main.py`.
*   `--host 0.0.0.0`: Cho phép truy cập từ mọi địa chỉ IP.
*   `--port 8000`: Chạy ứng dụng trên cổng 8000.
*   `--reload`: Tự động tải lại server khi có thay đổi mã nguồn (chỉ dùng khi phát triển).

API documentation sẽ có sẵn tại `http://localhost:8000/docs` hoặc `http://localhost:8000/redoc` sau khi server khởi động.

## Các API Endpoints

Hệ thống cung cấp các API sau:

### 1. Đăng ký khuôn mặt

**Endpoint:** `/register`
**Method:** `POST`
**Mô tả:** Đăng ký một khuôn mặt mới vào hệ thống cùng với thông tin người dùng.

**Parameters:**
*   `user_id` (string, query): ID duy nhất của người dùng.
*   `user_name` (string, query): Tên của người dùng.
*   `file` (file, form-data): Ảnh khuôn mặt của người dùng (JPEG/PNG).

**Ví dụ cURL:**

```bash
curl -X POST "http://localhost:8000/register?user_id=U001&user_name=NguyenVanA" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/face_image.jpg;type=image/jpeg"
```

### 2. Điểm danh

**Endpoint:** `/check_in`
**Method:** `POST`
**Mô tả:** Thực hiện điểm danh bằng cách nhận dạng khuôn mặt từ ảnh hoặc luồng video.

**Parameters:**
*   `file` (file, form-data): Ảnh chứa khuôn mặt cần điểm danh (JPEG/PNG).

**Ví dụ cURL:**

```bash
curl -X POST "http://localhost:8000/check_in" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/checkin_image.jpg;type=image/jpeg"
```

### 3. Lấy lịch sử điểm danh

**Endpoint:** `/get_attendance`
**Method:** `GET`
**Mô tả:** Lấy lịch sử điểm danh theo các tiêu chí lọc.

**Parameters:**
*   `date` (string, query, optional): Ngày cụ thể để lọc (định dạng YYYY-MM-DD).
*   `user_id` (string, query, optional): ID người dùng để lọc.

**Ví dụ cURL:**

```bash
# Lấy tất cả lịch sử điểm danh
curl -X GET "http://localhost:8000/get_attendance" -H "accept: application/json"

# Lấy lịch sử điểm danh theo ngày
curl -X GET "http://localhost:8000/get_attendance?date=2025-08-02" -H "accept: application/json"

# Lấy lịch sử điểm danh theo người dùng
curl -X GET "http://localhost:8000/get_attendance?user_id=U001" -H "accept: application/json"

# Lấy lịch sử điểm danh theo ngày và người dùng
curl -X GET "http://localhost:8000/get_attendance?date=2025-08-02&user_id=U001" -H "accept: application/json"
```

## Lưu ý quan trọng

*   **Mô hình YOLOv8:** Bạn cần tải xuống mô hình `yolov8n-face.pt` (hoặc tương tự) và đặt vào thư mục `models/`. Nếu bạn không tìm thấy mô hình `yolov8n-face.pt` cụ thể, bạn có thể sử dụng một mô hình YOLOv8 tổng quát và tự huấn luyện lại hoặc tìm các giải pháp thay thế cho phát hiện khuôn mặt.
*   **MobileFaceNet:** Module nhận dạng khuôn mặt sử dụng `InceptionResnetV1` từ thư viện `facenet_pytorch`, mô hình này sẽ được tự động tải xuống khi chạy lần đầu tiên.
*   **Vector Database:** FAISS được sử dụng để lưu trữ và tìm kiếm vector. Dữ liệu được lưu trữ trong thư mục `data/`.
*   **Lưu trữ điểm danh:** Lịch sử điểm danh hiện đang được lưu trữ trong bộ nhớ (biến `attendance_log`). Trong môi trường sản phẩm, bạn nên tích hợp một cơ sở dữ liệu bền vững (ví dụ: PostgreSQL, MySQL) để lưu trữ dữ liệu này.
*   **Xử lý lỗi:** Hệ thống có các cơ chế xử lý lỗi cơ bản. Bạn có thể mở rộng để xử lý các trường hợp lỗi phức tạp hơn.
*   **Bảo mật:** Đối với môi trường sản xuất, cần triển khai các biện pháp bảo mật bổ sung như xác thực người dùng, HTTPS, và kiểm soát truy cập.

## Tác giả

Manus AI



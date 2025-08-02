import cv2
import numpy as np
from ultralytics import YOLO

class FaceDetector:
    def __init__(self, model_path=\'yolov8n-face.pt\', confidence_threshold=0.5):
        # Load a pre-trained YOLOv8 model
        # For face detection, you might want to use a model specifically trained for faces
        # like \'yolov8n-face.pt\' if available, or fine-tune a generic YOLOv8n.
        # If yolov8n-face.pt is not available, you can use a general object detection model
        # like yolov8n.pt and filter for 'person' class, then apply a face detector on that.
        # For this example, we assume a face-specific YOLOv8 model.
        self.model = YOLO(model_path) # Load a pre-trained YOLOv8 model
        self.confidence_threshold = confidence_threshold

    def detect_faces(self, image):
        # Perform inference
        results = self.model(image, conf=self.confidence_threshold, verbose=False)

        face_boxes = []
        for r in results:
            # Assuming the model is trained to detect 'face' as a class
            # If using a generic YOLOv8, you might need to filter by class_id for 'person'
            # and then use another face detector (e.g., MTCNN) on the detected person regions.
            # For a face-specific YOLOv8 model, the detected objects are directly faces.
            boxes = r.boxes.xyxy.cpu().numpy() # xyxy format
            confs = r.boxes.conf.cpu().numpy()
            # class_ids = r.boxes.cls.cpu().numpy() # If you need to filter by class_id

            for i in range(len(boxes)):
                x1, y1, x2, y2 = map(int, boxes[i])
                conf = confs[i]
                # Assuming the model directly detects faces, no class_id check needed for 'face'
                # If using a general YOLOv8, you'd check class_ids here (e.g., if class_id == person_class_id)
                face_boxes.append([x1, y1, x2 - x1, y2 - y1, conf]) # Convert to x, y, w, h, conf
        return face_boxes

    def extract_face_from_image(self, image, box):
        x, y, w, h, _ = box
        # Ensure coordinates are within image bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)
        face_image = image[y:y+h, x:x+w]
        return face_image

# Example usage (for testing purposes, not part of the main API)
if __name__ == \'__main__\':
    # Download yolov8n-face.pt or similar pre-trained YOLOv8 face model
    # You can find models on Hugging Face or Ultralytics GitHub.
    # For example, from https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5n.pt (general object detection)
    # or a specific face detection model if available.
    # For this example, we assume \'yolov8n-face.pt\' exists in the models directory.
    model_path = \'./models/yolov8n-face.pt\'

    # Create a dummy model file for demonstration if it doesn\'t exist
    # In a real scenario, you would download the actual .pt file.
    if not os.path.exists(model_path):
        print(f"Warning: Model file {model_path} not found. Creating a dummy file. Please download the actual YOLOv8 face model.")
        # This is a placeholder. You need to download the actual model.
        # For example, you can download yolov8n.pt from Ultralytics and rename it
        # or find a specific yolov8-face.pt model.
        with open(model_path, \'w\') as f:
            f.write(\'# Dummy YOLOv8 face model file\n\')

    detector = FaceDetector(model_path=model_path)

    # Create a dummy image for testing
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(dummy_image, \"Dummy Image\", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Simulate a detected face (for testing purposes)
    # In a real scenario, detect_faces would return actual boxes
    simulated_face_box = [100, 100, 200, 200, 0.9] # x, y, w, h, confidence
    # Draw a rectangle to simulate a face
    cv2.rectangle(dummy_image, (simulated_face_box[0], simulated_face_box[1]),
                  (simulated_face_box[0] + simulated_face_box[2], simulated_face_box[1] + simulated_face_box[3]),
                  (0, 255, 0), 2)

    # Test detect_faces (will return empty if no real model is loaded or no faces detected)
    # For actual testing, you'd pass a real image with faces.
    print(\"Attempting to detect faces with dummy image (requires real model for actual detection)...\")
    detected_faces = detector.detect_faces(dummy_image)
    print(f\"Detected faces: {detected_faces}\")

    # Test extract_face_from_image
    if simulated_face_box:
        extracted_face = detector.extract_face_from_image(dummy_image, simulated_face_box)
        if extracted_face is not None and extracted_face.size > 0:
            print(f\"Extracted face image shape: {extracted_face.shape}\")
            # You can save or display the extracted face for verification
            # cv2.imwrite(\"extracted_face.jpg\", extracted_face)
        else:
            print(\"Failed to extract face or extracted face is empty.\")
    else:
        print(\"No simulated face box to extract.\")



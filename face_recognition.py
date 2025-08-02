import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch

class FaceRecognizer:
    def __init__(self, model_path=None):
        # Initialize MTCNN for face detection and alignment (optional, but good for robust embeddings)
        # We can use it here to ensure the face image is properly aligned before feeding to MobileFaceNet
        # However, since YOLOv8 is already detecting faces, we will primarily use it for embedding generation.
        # If you want MTCNN for alignment, you can uncomment and use it.
        # self.mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, device=\'cuda\' if torch.cuda.is_available() else \'cpu\')

        # Load a pre-trained InceptionResnetV1 model (often used as a backbone for FaceNet)
        # This model is pre-trained on VGGFace2 and CASIA-Webface datasets.
        # It outputs 512-dimensional embeddings.
        self.model = InceptionResnetV1(pretrained=\'vggface2\').eval() # .eval() sets the model to evaluation mode
        self.device = torch.device(\'cuda\' if torch.cuda.is_available() else \'cpu\')
        self.model.to(self.device)
        print(f"MobileFaceNet (InceptionResnetV1) model loaded on {self.device}.")

    def get_face_embedding(self, face_image):
        if face_image is None or face_image.size == 0:
            return None

        # Convert OpenCV image (numpy array) to PIL Image
        face_image_pil = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))

        # Preprocess the face image for the model
        # The InceptionResnetV1 model expects input images of size 160x160
        # and normalized to [-1, 1].
        # facenet_pytorch handles this preprocessing internally if you pass PIL image.
        
        # If you were using MTCNN for alignment:
        # face_aligned = self.mtcnn(face_image_pil)
        # if face_aligned is None:
        #     return None
        # face_tensor = face_aligned.to(self.device)

        # Without MTCNN alignment, just resize and convert to tensor
        # Resize to 160x160 and convert to tensor, then normalize
        # This part is crucial for correct embedding generation.
        # The `InceptionResnetV1` model expects input in the format of `(batch_size, channels, height, width)`
        # and pixel values normalized to [-1, 1].
        # `facenet_pytorch` has internal transforms for this when `model.eval()` is used with PIL images.
        
        # Simple resize and convert to tensor, then normalize manually if needed
        # For `InceptionResnetV1`, the `transform` applied during `__init__` handles this.
        # We just need to ensure the input is a PIL Image and then convert to tensor.
        
        # Convert PIL image to tensor and move to device
        # The model's forward pass expects a tensor, and it applies its own preprocessing.
        # We need to ensure the input is a tensor of shape (1, 3, 160, 160) and type float32.
        
        # Simple resize and convert to tensor
        face_image_resized = cv2.resize(face_image, (160, 160))
        face_image_tensor = torch.from_numpy(face_image_resized).permute(2, 0, 1).float() # HWC to CHW, to float
        face_image_tensor = (face_image_tensor / 255.0 - 0.5) * 2.0 # Normalize to [-1, 1]
        face_image_tensor = face_image_tensor.unsqueeze(0).to(self.device) # Add batch dimension

        with torch.no_grad(): # Disable gradient calculation for inference
            embedding = self.model(face_image_tensor).cpu().numpy() # Get embedding and move to CPU
        
        return embedding.flatten() # Return a 1D numpy array

    def compare_embeddings(self, embedding1, embedding2, threshold=0.7):
        # Use cosine similarity to compare two face embeddings
        if embedding1 is None or embedding2 is None:
            return False, 0.0
        
        embedding1 = embedding1.reshape(1, -1)
        embedding2 = embedding2.reshape(1, -1)
        
        similarity = cosine_similarity(embedding1, embedding2)[0][0]
        return similarity >= threshold, similarity

# Example usage (for testing purposes, not part of the main API)
if __name__ == \'__main__\':
    recognizer = FaceRecognizer()

    # Create dummy face images (simulating extracted faces)
    dummy_face1 = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
    dummy_face2 = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
    dummy_face3 = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)

    # Get embeddings
    embedding1 = recognizer.get_face_embedding(dummy_face1)
    embedding2 = recognizer.get_face_embedding(dummy_face2)
    embedding3 = recognizer.get_face_embedding(dummy_face3)

    if embedding1 is not None:
        print(f"Embedding 1 shape: {embedding1.shape}")
    if embedding2 is not None:
        print(f"Embedding 2 shape: {embedding2.shape}")

    # Compare embeddings
    if embedding1 is not None and embedding2 is not None:
        is_same, similarity = recognizer.compare_embeddings(embedding1, embedding2, threshold=0.7)
        print(f"Comparison between embedding 1 and 2: Is same = {is_same}, Similarity = {similarity:.4f}")

    # Simulate a very similar embedding for testing
    # In a real scenario, you'd get two actual embeddings of the same person.
    # For simulation, let's assume embedding1 is from person A, and we create a slightly perturbed version.
    if embedding1 is not None:
        similar_embedding1 = embedding1 + np.random.rand(embedding1.shape[0]) * 0.01 # Add small noise
        is_same_sim, similarity_sim = recognizer.compare_embeddings(embedding1, similar_embedding1, threshold=0.7)
        print(f"Comparison between embedding 1 and similar 1: Is same = {is_same_sim}, Similarity = {similarity_sim:.4f}")

    # Test with None embedding
    is_same_none, similarity_none = recognizer.compare_embeddings(embedding1, None)
    print(f"Comparison with None embedding: Is same = {is_same_none}, Similarity = {similarity_none:.4f}")



import faiss
import numpy as np
import json
import os

class VectorDB:
    def __init__(self, dimension=512, db_path="./data/face_embeddings.bin", id_map_path="./data/id_map.json"):
        self.dimension = dimension
        self.db_path = db_path
        self.id_map_path = id_map_path
        self.index = None
        self.id_map = {}
        self._load_db()

    def _load_db(self):
        if os.path.exists(self.db_path):
            self.index = faiss.read_index(self.db_path)
            print(f"Loaded FAISS index from {self.db_path}")
        else:
            # Using IndexFlatL2 for simple Euclidean distance search
            # For larger datasets, consider IndexIVFFlat or IndexHNSWFlat
            self.index = faiss.IndexFlatL2(self.dimension)
            print("Created new FAISS index.")

        if os.path.exists(self.id_map_path):
            with open(self.id_map_path, \'r\') as f:
                self.id_map = json.load(f)
            print(f"Loaded ID map from {self.id_map_path}")
        else:
            print("Created new ID map.")

    def _save_db(self):
        faiss.write_index(self.index, self.db_path)
        with open(self.id_map_path, \'w\') as f:
            json.dump(self.id_map, f)
        print(f"Saved FAISS index to {self.db_path} and ID map to {self.id_map_path}")

    def add_face(self, user_id: str, user_name: str, embedding: np.ndarray):
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1) # Ensure it\'s 2D
        
        if embedding.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.dimension}, got {embedding.shape[1]}")

        # Add the embedding to the FAISS index
        self.index.add(embedding)
        
        # Store the mapping from FAISS internal ID to user_id and user_name
        # FAISS internal IDs are sequential, starting from 0
        faiss_id = self.index.ntotal - 1 # Get the ID of the newly added vector
        self.id_map[str(faiss_id)] = {\"user_id\": user_id, \"user_name\": user_name}
        self._save_db()
        print(f"Added face for user {user_name} (ID: {user_id}) with FAISS ID {faiss_id}")
        return faiss_id

    def search_face(self, query_embedding: np.ndarray, k: int = 1, threshold: float = 0.6):
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1) # Ensure it\'s 2D
        
        if query_embedding.shape[1] != self.dimension:
            raise ValueError(f"Query embedding dimension mismatch. Expected {self.dimension}, got {query_embedding.shape[1]}")

        # Perform similarity search
        distances, indices = self.index.search(query_embedding, k) # D is distances, I is indices

        results = []
        for i in range(k):
            faiss_id = indices[0][i]
            distance = distances[0][i]
            
            # Convert distance to similarity (for L2 distance, smaller is better, so 1 / (1 + distance) or similar)
            # For cosine similarity, you\'d typically normalize embeddings and use dot product.
            # Since we used IndexFlatL2, distance is Euclidean. We need to convert it to a similarity score.
            # A common way is to use a threshold on distance directly, or convert to a [0,1] range.
            # A simple conversion to a similarity score (higher is better, max 1.0 for identical)
            # This is a heuristic, adjust based on your embedding space and desired behavior.
            similarity = 1.0 / (1.0 + distance) # Example: 0 distance -> 1.0 similarity, large distance -> near 0 similarity

            if str(faiss_id) in self.id_map and similarity >= threshold:
                user_info = self.id_map[str(faiss_id)]
                results.append({
                    \"user_id\": user_info[\"user_id\"],
                    \"user_name\": user_info[\"user_name\"],
                    \"similarity\": similarity,
                    \"faiss_id\": faiss_id
                })
            elif str(faiss_id) in self.id_map and similarity < threshold:
                print(f\"Found match for FAISS ID {faiss_id} but similarity {similarity:.4f} is below threshold {threshold}\")
            else:
                print(f\"FAISS ID {faiss_id} not found in id_map or no match above threshold.\")
        return results

    def get_total_faces(self):
        return self.index.ntotal

# Example usage (for testing purposes)
if __name__ == \'__main__\':
    # Ensure the data directory exists
    os.makedirs(\"./data\", exist_ok=True)

    db = VectorDB(dimension=512) # Assuming MobileFaceNet produces 512-dim embeddings

    # Add some dummy faces
    embedding1 = np.random.rand(512).astype(np.float32)
    db.add_face(\"user001\", \"Alice\", embedding1)

    embedding2 = np.random.rand(512).astype(np.float32)
    db.add_face(\"user002\", \"Bob\", embedding2)

    # Simulate a new face (query embedding) that is very similar to Alice
    query_embedding_alice = embedding1 + np.random.rand(512).astype(np.float32) * 0.01 # Small noise
    
    # Simulate a new face (query embedding) that is very similar to Bob
    query_embedding_bob = embedding2 + np.random.rand(512).astype(np.float32) * 0.01 # Small noise

    # Simulate a new face (query embedding) that is not similar to anyone
    query_embedding_unknown = np.random.rand(512).astype(np.float32)

    print(\"\\nSearching for Alice...\")
    results_alice = db.search_face(query_embedding_alice, k=1, threshold=0.9) # Higher threshold for close match
    if results_alice:
        print(f\"Found: {results_alice[0][\"user_name\"]} with similarity {results_alice[0][\"similarity\"]:.4f}\")
    else:
        print(\"Alice not found or similarity too low.\")

    print(\"\\nSearching for Bob...\")
    results_bob = db.search_face(query_embedding_bob, k=1, threshold=0.9)
    if results_bob:
        print(f\"Found: {results_bob[0][\"user_name\"]} with similarity {results_bob[0][\"similarity\"]:.4f}\")
    else:
        print(\"Bob not found or similarity too low.\")

    print(\"\\nSearching for unknown face...\")
    results_unknown = db.search_face(query_embedding_unknown, k=1, threshold=0.9)
    if results_unknown:
        print(f\"Found: {results_unknown[0][\"user_name\"]} with similarity {results_unknown[0][\"similarity\"]:.4f}\")
    else:
        print(\"Unknown face not found or similarity too low.\")

    print(f\"Total faces in DB: {db.get_total_faces()}\")

    # Clean up dummy files after testing
    # os.remove(\"./data/face_embeddings.bin\")
    # os.remove(\"./data/id_map.json\")
    # os.rmdir(\"./data\")



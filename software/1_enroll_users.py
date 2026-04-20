import os
import cv2
import faiss
import numpy as np
import pickle
from insightface.app import FaceAnalysis

# 1. Initialize the SOTA AI (RetinaFace + ArcFace)
print("[INFO] Loading AI Models... (This might take a minute to download weights the first time)")
app = FaceAnalysis(name='buffalo_l') 
app.prepare(ctx_id=0, det_size=(640, 640)) # ctx_id=0 means we use CPU

DATASET_PATH = "dataset"
VECTOR_DIMENSION = 512 # ArcFace outputs 512 numbers per face

# 2. Initialize FAISS Database (Inner Product + Normalized Vectors = Cosine Similarity)
index = faiss.IndexFlatIP(VECTOR_DIMENSION) 
user_mapping = {} # Maps Faiss ID to the person's name
current_id = 0

print("[INFO] Scanning dataset and extracting 512-d vectors...")

# Check if dataset exists
if not os.path.exists(DATASET_PATH):
    print(f"[ERROR] '{DATASET_PATH}' folder not found!")
    exit()

# 3. Process the photos
for person_name in os.listdir(DATASET_PATH):
    person_dir = os.path.join(DATASET_PATH, person_name)
    
    # Skip standard files or the "unknown" folder (we don't save strangers!)
    if not os.path.isdir(person_dir) or person_name == "unknown":
        continue 
        
    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            continue
            
        # The AI does everything here: Detects, aligns, and extracts the fingerprint
        faces = app.get(img)
        
        if len(faces) == 0:
            print(f"[WARN] No face found in {img_name}. Skipping.")
            continue
            
        # Take the embedding of the largest/first face found
        embedding = faces[0].embedding
        
        # Normalize the vector (Critical for strict Cosine Similarity math)
        embedding = embedding.reshape(1, -1)
        faiss.normalize_L2(embedding)
        
        # Add to FAISS Database
        index.add(embedding)
        user_mapping[current_id] = person_name
        current_id += 1
        print(f"[+] Saved fingerprint for {person_name} ({img_name})")

# 4. Save the Database to disk
faiss.write_index(index, "data/vector_database.index")
with open("data/user_mapping.pkl", "wb") as f:
    pickle.dump(user_mapping, f)

print(f"\n✅ Enterprise Database Built! Saved {current_id} faces.")
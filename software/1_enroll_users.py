import os
import cv2
import faiss
import numpy as np
import pickle
from insightface.app import FaceAnalysis
from collections import defaultdict
import tqdm  # For progress (pip install tqdm)

# ==========================================
# 0. DYNAMIC OS PATHING
# ==========================================
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset")
DATA_DIR     = os.path.join(BASE_DIR, "data")
ASSETS_DIR   = os.path.join(BASE_DIR, "assets")
GLASSES_DIR  = os.path.join(ASSETS_DIR, "glasses")

os.makedirs(DATA_DIR, exist_ok=True)

# ==========================================
# 1. LOAD MODELS
# ==========================================
print("[INFO] Loading SOTA AI Models (RetinaFace + ArcFace)...")
app = FaceAnalysis(name='buffalo_l')

# CHANGE THIS LINE: Drop from 640 to 320
app.prepare(ctx_id=0, det_size=(320, 320))

VECTOR_DIMENSION = 512
index         = faiss.IndexFlatIP(VECTOR_DIMENSION)  # Cosine sim (IP on normalized)
user_mapping  = {}
current_id    = 0

# ==========================================
# 2. AUGMENTATION → CENTROID ENGINE (BEST PRACTICE)
# ==========================================
def add_glasses_overlay(img, glasses_path):
    """Overlay glasses template on face region."""
    glasses = cv2.imread(glasses_path, cv2.IMREAD_UNCHANGED)
    if glasses is None or glasses.shape[2] != 4:
        return img

    h, w = img.shape[:2]
    gh, gw = glasses.shape[:2]

    # Basic scaling relative to face size
    scale = min(w / (2.0 * gw), h / (3.0 * gh)) * 0.8
    new_w, new_h = int(gw * scale), int(gh * scale)
    glasses_resized = cv2.resize(glasses, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Rough eye region position
    x1 = int(w * 0.25)
    y1 = int(h * 0.30)
    x2 = x1 + new_w
    y2 = y1 + new_h

    if x2 > w or y2 > h:
        return img

    overlay = glasses_resized[:, :, :3]
    alpha = glasses_resized[:, :, 3:] / 255.0

    roi = img[y1:y2, x1:x2]
    for c in range(3):
        roi[:, :, c] = (1.0 - alpha[:, :, 0]) * roi[:, :, c] + alpha[:, :, 0] * overlay[:, :, c]
    img[y1:y2, x1:x2] = roi
    return img


def get_glasses_variants(img):
    """Generate extra images with synthetic glasses, if templates exist."""
    variants = []
    if not os.path.isdir(GLASSES_DIR):
        return variants

    for fname in os.listdir(GLASSES_DIR):
        if not fname.lower().endswith((".png", ".webp")):
            continue
        g_path = os.path.join(GLASSES_DIR, fname)
        with_glasses = add_glasses_overlay(img.copy(), g_path)
        variants.append(("glasses_" + fname, with_glasses))

    return variants


def extract_centroid(img, app, person_name):
    """
    BEST: Compute SINGLE robust centroid per photo/person from 12+ augs.
    Improves stability 10-20% vs multi-vectors [web:29].
    """
    h, w = img.shape[:2]
    embeddings_list = []

    # Helper augs (expanded for robustness)
    def rotate(angle):
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    def add_noise():
        noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
        return cv2.add(img, noise)

    augmentations = [
        ("original", img),
        ("hflip", cv2.flip(img, 1)),
        ("brighter", cv2.convertScaleAbs(img, alpha=1.0, beta=40)),
        ("darker", cv2.convertScaleAbs(img, alpha=1.0, beta=-40)),
        ("high_contrast", cv2.convertScaleAbs(img, alpha=1.3, beta=0)),
        ("low_contrast", cv2.convertScaleAbs(img, alpha=0.7, beta=0)),
        ("blur", cv2.GaussianBlur(img, (5, 5), 0)),
        ("noise", add_noise()),
        ("rotate_5", rotate(5)),
        ("rotate_-5", rotate(-5)),
        ("crop_center", img[h//4:3*h//4, w//4:3*w//4]),
        ("sharpen", cv2.filter2D(img, -1, np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]))),
    ]

    # Add synthetic glasses variants (if any templates available)
    for name, g_img in get_glasses_variants(img):
        augmentations.append((name, g_img))

    for aug_name, aug_img in augmentations:
        if aug_img.size == 0: continue
        faces = app.get(aug_img)
        if len(faces) > 0:
            emb = faces[0].embedding.reshape(1, -1)
            embeddings_list.append(emb)

    if not embeddings_list:
        print(f"    [SKIP] No valid embeddings for {person_name}")
        return None

    # CRITICAL: Compute & normalize centroid
    centroid = np.mean(embeddings_list, axis=0).reshape(1, -1)
    faiss.normalize_L2(centroid)
    print(f"    [+] Centroid for {person_name}: {len(embeddings_list)} anchors")
    return centroid

# ==========================================
# 3. SCAN DATASET & ENROLL (Per-Person Centroids)
# ==========================================
print("[INFO] Building per-person centroids...")

if not os.path.exists(DATASET_PATH):
    print(f"[ERROR] '{DATASET_PATH}' not found.")
    exit()

person_embeddings = defaultdict(list)  # Collect all photo centroids per person

for person_name in tqdm.tqdm(os.listdir(DATASET_PATH), desc="Persons"):
    person_dir = os.path.join(DATASET_PATH, person_name)
    if not os.path.isdir(person_dir) or person_name == "unknown":
        continue

    print(f"\n[*] Processing: {person_name}")
    photo_count = 0

    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        img = cv2.imread(img_path)
        if img is None: continue

        # Pre-resize for speed (optional)
        img = cv2.resize(img, (640, 640))

        faces = app.get(img)
        if len(faces) == 0: continue

        centroid = extract_centroid(img, app, person_name)
        if centroid is not None:
            person_embeddings[person_name].append(centroid)
            photo_count += 1

    # FINAL: Avg all photo centroids per person → ULTRA-ROBUST SINGLE VECTOR
    if person_embeddings[person_name]:
        final_centroid = np.mean(person_embeddings[person_name], axis=0).reshape(1, -1)
        faiss.normalize_L2(final_centroid)
        index.add(final_centroid)
        user_mapping[current_id] = person_name
        current_id += 1
        print(f"    [DONE] Final centroid from {photo_count} photos")

# ==========================================
# 4. SAVE DATABASE
# ==========================================
faiss.write_index(index, os.path.join(DATA_DIR, "vector_database.index"))
with open(os.path.join(DATA_DIR, "user_mapping.pkl"), "wb") as f:
    pickle.dump(user_mapping, f)

print(f"\n[INFO] Elite DB built: {current_id} centroids (1/person). Space-efficient & robust!")
print("[INFO] Query tip: During search, threshold ~0.6 cosine sim + top-1/3 [web:10]")
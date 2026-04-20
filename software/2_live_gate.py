import cv2
import faiss
import numpy as np
import pickle
import threading
import time
import os
import psycopg2
import onnxruntime as ort
from insightface.app import FaceAnalysis

# ==========================================
# 1. VARIABLES PARTAGEES ET COOLDOWN
# ==========================================
latest_frame = None
current_detections = []
lock = threading.Lock()
running = True

last_log_time = {}
LOG_COOLDOWN = 5.0 

# ==========================================
# 2. CONNEXION POSTGRESQL (ETL)
# ==========================================
print("[INFO] Initialisation de la connexion PostgreSQL...")
try:
    db_conn = psycopg2.connect(
        dbname="sentinel_logs_db",
        user="postgres",
        password="youssef", # METS TON MOT DE PASSE ICI
        host="127.0.0.1",
        port="5432"
    )
    db_conn.autocommit = True
    db_cursor = db_conn.cursor()
    print("[INFO] Connexion au serveur de base de donnees etablie avec succes.")
except psycopg2.Error as e:
    print(f"[ERREUR CRITIQUE] Echec de la connexion PostgreSQL: {e}")
    exit(1)

def log_to_postgres(name, status, liveness, confidence):
    current_time = time.time()
    
    if name in last_log_time and (current_time - last_log_time[name]) < LOG_COOLDOWN:
        return 
        
    last_log_time[name] = current_time
    
    try:
        insert_query = """
            INSERT INTO access_logs (person_name, status, liveness_score, confidence_score)
            VALUES (%s, %s, %s, %s)
        """
        db_cursor.execute(insert_query, (name, status, float(liveness), float(confidence)))
        print(f"[DB LOG] Requete inseree: {name} | {status} | Liveness: {liveness*100:.1f}%")
    except psycopg2.Error as e:
        print(f"[ERREUR DB] {e}")

# ==========================================
# 3. INITIALISATION AI & FAS
# ==========================================
print("[INFO] Chargement du modele InsightFace buffalo_l...")
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640)) 

index = faiss.read_index("vector_database.index")
with open("user_mapping.pkl", "rb") as f:
    user_mapping = pickle.load(f)

FAS_MODEL_PATH = "fas_model.onnx"
fas_session = None

if os.path.exists(FAS_MODEL_PATH):
    print("[INFO] Module Anti-Spoofing MiniFASNet charge.")
    fas_session = ort.InferenceSession(FAS_MODEL_PATH, providers=['CPUExecutionProvider'])
else:
    print("[AVERTISSEMENT] fas_model.onnx introuvable. Systeme Liveness desactive.")

RECOGNITION_THRESHOLD = 0.70 
FAS_REAL_THRESHOLD = 0.5 

# ==========================================
# 4. MOTEUR IA ASYNCHRONE
# ==========================================
def preprocess_fas_image(face_img):
    # 1. Resize au format du modele
    img = cv2.resize(face_img, (112, 112))
    # 2. Conversion de BGR (OpenCV) vers RGB (Standard AI)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 3. Normalisation des pixels de [0, 255] vers [0.0, 1.0] (L-Correction l-asasiya)
    img = np.float32(img) / 255.0
    # 4. Transposition en format Channel-Height-Width (CHW)
    img = img.transpose((2, 0, 1)) 
    img = np.expand_dims(img, axis=0) 
    return img

def ai_worker():
    global latest_frame, current_detections, running
    
    while running:
        with lock:
            if latest_frame is None:
                frame_to_process = None
            else:
                frame_to_process = latest_frame.copy()
        
        if frame_to_process is None:
            time.sleep(0.01)
            continue
            
        faces = app.get(frame_to_process)
        new_detections = []
        
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            
            is_real_face = True
            liveness_score = 1.0
            
            if fas_session is not None:
                h_img, w_img = frame_to_process.shape[:2]
                margin = int((y2 - y1) * 0.3)
                crop_x1 = max(0, x1 - margin)
                crop_y1 = max(0, y1 - margin)
                crop_x2 = min(w_img, x2 + margin)
                crop_y2 = min(h_img, y2 + margin)
                
                face_roi = frame_to_process[crop_y1:crop_y2, crop_x1:crop_x2]
                
                if face_roi.size > 0:
                    try:
                        fas_input = preprocess_fas_image(face_roi)
                        ort_inputs = {fas_session.get_inputs()[0].name: fas_input}
                        preds = fas_session.run(None, ort_inputs)[0]
                        exp_preds = np.exp(preds[0])
                        softmax_preds = exp_preds / np.sum(exp_preds)
                        
                        liveness_score = softmax_preds[1] 
                        if liveness_score < FAS_REAL_THRESHOLD:
                            is_real_face = False
                    except Exception:
                        pass
            
            if face.kps is not None and len(face.kps) >= 2:
                eye_dist = np.linalg.norm(face.kps[0] - face.kps[1])
            else:
                eye_dist = 0
            
            live_embedding = face.embedding.reshape(1, -1)
            faiss.normalize_L2(live_embedding)

            distances, indices = index.search(live_embedding, k=1)
            confidence = distances[0][0]
            
            detected_name = user_mapping.get(indices[0][0], "Unknown")
            
            if not is_real_face:
                color = (0, 165, 255) 
                status = f"SPOOF DETECTED! ({int(liveness_score*100)}%)"
                log_to_postgres("IMPOSTER", "SPOOF_ATTEMPT", liveness_score, confidence)
            elif eye_dist < 40:
                color = (50, 50, 255) 
                status = "TOO FAR"
            elif confidence > RECOGNITION_THRESHOLD:
                color = (0, 255, 100) 
                status = f"GRANTED | {detected_name.upper()}"
                log_to_postgres(detected_name, "ACCESS_GRANTED", liveness_score, confidence)
            else:
                color = (50, 50, 255) 
                status = "ACCESS DENIED"
                log_to_postgres("STRANGER", "ACCESS_DENIED", liveness_score, confidence)
                
            new_detections.append({
                'bbox': bbox, 
                'status': status, 
                'color': color
            })
            
        with lock:
            current_detections = new_detections

worker_thread = threading.Thread(target=ai_worker, daemon=True)
worker_thread.start()

# ==========================================
# 5. INTERFACE VIDEO
# ==========================================
# Remplacez l'adresse IP par celle affichée sur votre téléphone !
cap = cv2.VideoCapture("http://192.168.1.22:8080/video")
print("[INFO] Sentinel V2 Pipeline Operationnel. Appuyez sur 'q' pour fermer.")

while True:
    ret, frame = cap.read()
    if not ret: break
    # N-dewro l-vidéo b 90 daraja bach t-ban m-qadda f l-Portrait
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    # 2. N-sghro l-cadre bach y-ban mzyan f l-ecran (Largeur: 450, Hauteur: 800)
    frame = cv2.resize(frame, (450, 800))

    with lock:
        latest_frame = frame
        detections_to_draw = current_detections.copy()

    for det in detections_to_draw:
        x1, y1, x2, y2 = det['bbox']
        color = det['color']
        status = det['status']
        
        length, thickness = 25, 2
        cv2.line(frame, (x1, y1), (x1 + length, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + length), color, thickness)
        cv2.line(frame, (x2, y1), (x2 - length, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + length), color, thickness)
        cv2.line(frame, (x1, y2), (x1 + length, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - length), color, thickness)
        cv2.line(frame, (x2, y2), (x2 - length, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - length), color, thickness)

        cv2.rectangle(frame, (x1, y1 - 35), (x1 + len(status)*10, y1 - 10), color, cv2.FILLED)
        text_color = (0, 0, 0) if color == (0, 255, 100) else (255, 255, 255)
        cv2.putText(frame, status, (x1 + 5, y1 - 20), cv2.FONT_HERSHEY_DUPLEX, 0.45, text_color, 1)

    cv2.imshow("SENTINEL V2", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        running = False
        break

cap.release()
cv2.destroyAllWindows()
db_cursor.close()
db_conn.close()
worker_thread.join()
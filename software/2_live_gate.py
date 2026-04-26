import cv2
import faiss
import numpy as np
import pickle
import threading
import time
import os
import sqlite3
import requests
import onnxruntime as ort
import multiprocessing as mp
from insightface.app import FaceAnalysis

# ==========================================
# 0. DYNAMIC PATHS & CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── ESP32-CAM Endpoints ──────────────────────────────────────────────────────
ESP32_STREAM_URL = "http://192.168.43.178:81/stream"
ESP32_GRANT_URL  = "http://192.168.43.178/access_granted"
ESP32_DENY_URL   = "http://192.168.43.178/access_denied"
ESP32_HEALTH_URL = "http://192.168.43.178/health"
ESP32_TIMEOUT    = 1.5   # seconds — non-blocking, gate should not wait

BASE_RECOGNITION_THRESHOLD = 0.55
FAS_REAL_THRESHOLD         = 0.40
EYE_DIST_MIN               = 40
EYE_DIST_MAX               = 220
EYE_AR_THRESH              = 0.28

# ==========================================
# 1. HELPER CLASSES & FUNCTIONS
# ==========================================
class BoxSmoother:
    def __init__(self, alpha=0.4):
        self.smooth_bbox = None
        self.alpha       = alpha

    def update(self, new_bbox):
        if self.smooth_bbox is None:
            self.smooth_bbox = np.array(new_bbox, dtype=float)
        else:
            self.smooth_bbox = (np.array(new_bbox) * self.alpha + self.smooth_bbox * (1 - self.alpha))
        return self.smooth_bbox.astype(int)

class CameraStream:
    """Non-blocking webcam capture using a background thread."""
    def __init__(self, src):
        if isinstance(src, int):
            self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)  # Windows DirectShow
        else:
            self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = False, None
        for _ in range(30):  # warm-up retries
            self.ret, self.frame = self.cap.read()
            if self.ret and self.frame is not None:
                break
            time.sleep(0.1)
        if not self.ret:
            print("[ERROR] Cannot open camera. Check that nothing else is using it.")
        self._lock = threading.Lock()
        self._stop = threading.Event()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while not self._stop.is_set():
            ret, frame = self.cap.read()
            with self._lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self._lock:
            if self.ret and self.frame is not None:
                return True, self.frame.copy()
            return False, None

    def release(self):
        self._stop.set()
        self.cap.release()

def calculate_ear_native(landmarks):
    left_v1 = np.linalg.norm(landmarks[37][:2] - landmarks[41][:2])
    left_v2 = np.linalg.norm(landmarks[38][:2] - landmarks[40][:2])
    left_h  = np.linalg.norm(landmarks[36][:2] - landmarks[39][:2])
    ear_left = (left_v1 + left_v2) / (2.0 * left_h) if left_h != 0 else 0

    right_v1 = np.linalg.norm(landmarks[43][:2] - landmarks[47][:2])
    right_v2 = np.linalg.norm(landmarks[44][:2] - landmarks[46][:2])
    right_h  = np.linalg.norm(landmarks[42][:2] - landmarks[45][:2])
    ear_right = (right_v1 + right_v2) / (2.0 * right_h) if right_h != 0 else 0

    return (ear_left + ear_right) / 2.0

def preprocess_fas_image(face_img):
    """Reverted to standard scaling. Do NOT use ImageNet Mean/Std for this specific ONNX."""
    img = cv2.resize(face_img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.float32(img) / 255.0
    img = img.transpose((2, 0, 1))
    return np.expand_dims(img, axis=0)

def stable_softmax(x):
    shifted  = x - np.max(x)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals)

# ==========================================
# 2. AI WORKER PROCESS (ISOLATED)
# ==========================================
def ai_worker(frame_queue, result_queue):
    print("[INFO] Worker: Connecting to SQLite tickets.db...")
    last_log_time = {}
    LOG_COOLDOWN  = 5.0

    # ── SQLite (shared with server.py) ───────────────────────────────────────
    _db_path = os.path.join(BASE_DIR, "data", "tickets.db")
    try:
        db_conn = sqlite3.connect(_db_path, check_same_thread=False)
        db_conn.row_factory = sqlite3.Row
        db_conn.execute("""
            CREATE TABLE IF NOT EXISTS access_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_name TEXT,
                status TEXT,
                liveness_score REAL,
                confidence_score REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        db_conn.commit()
        print("[INFO] Worker: SQLite connected.")
    except Exception as e:
        print(f"[ERROR] SQLite: {e}")
        db_conn = None

    def log_to_sqlite(name, status, liveness, confidence):
        if not db_conn: return
        current_time = time.time()
        if name in last_log_time and (current_time - last_log_time[name]) < LOG_COOLDOWN: return
        last_log_time[name] = current_time
        try:
            db_conn.execute(
                "INSERT INTO access_logs (person_name, status, liveness_score, confidence_score) VALUES (?, ?, ?, ?)",
                (name, status, float(liveness), float(confidence))
            )
            db_conn.commit()
        except Exception:
            pass

    def signal_esp32(url):
        """Fire-and-forget HTTP signal to ESP32. Non-blocking."""
        try:
            requests.get(url, timeout=ESP32_TIMEOUT)
        except Exception:
            pass  # Gate signal failure must never crash the recognition loop

    # ── Check ESP32 health at startup ────────────────────────────────────────
    try:
        r = requests.get(ESP32_HEALTH_URL, timeout=2)
        print(f"[INFO] ESP32 health: {r.text.strip()}")
    except Exception:
        print("[WARNING] ESP32 not reachable — gate signals will be skipped.")

    print("[INFO] Worker: Initialisation des modeles AI...")
    app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'recognition', 'landmark_3d_68'])
    app.prepare(ctx_id=0, det_size=(320, 320))

    index = faiss.read_index(os.path.join(BASE_DIR, "data", "vector_database.index"))
    with open(os.path.join(BASE_DIR, "data", "user_mapping.pkl"), "rb") as f:
        user_mapping = pickle.load(f)

    FAS_PATH = os.path.join(BASE_DIR, "models", "fas_model.onnx")
    fas_session = ort.InferenceSession(FAS_PATH, providers=['CPUExecutionProvider']) if os.path.exists(FAS_PATH) else None

    ui_smoother = BoxSmoother(alpha=0.4)
    
    # State Dictionaries
    smoothed_liveness_state = {}
    smoothed_ear_state      = {}
    blink_state             = {}
    
    EMA_FAS_ALPHA = 0.3
    EMA_EAR_ALPHA = 0.4
    # ---------------------------------------------------
    # THE WARM-UP FIX: Pre-allocate memory buffers
    # ---------------------------------------------------
    print("[INFO] Worker: Pre-chauffage des modeles (Warm-up)...")
    dummy_frame = np.zeros((320, 320, 3), dtype=np.uint8)
    _ = app.get(dummy_frame)  # Forces InsightFace to initialize graphs
    
    if fas_session is not None:
        dummy_fas = np.zeros((1, 3, 112, 112), dtype=np.float32)
        _ = fas_session.run(None, {fas_session.get_inputs()[0].name: dummy_fas}) # Forces ONNX to initialize
    print("[INFO] Worker: Warm-up termine. Pret pour le temps reel.")
    # ---------------------------------------------------
    while True:
        frame_to_process = frame_queue.get()
        if frame_to_process is None: break
        # ---------------------------------------------------
        # THE LATENCY FIX: Drain the queue of stale frames
        # ---------------------------------------------------
        while not frame_queue.empty():
            try:
                dropped_frame = frame_queue.get_nowait()
                if dropped_frame is None:
                    return  # Catch shutdown signal during drain
                frame_to_process = dropped_frame  # Always overwrite with the newest
            except Exception:
                pass
        # ---------------------------------------------------
        faces = app.get(frame_to_process)
        new_detections = []

        if len(faces) != 1:
            ui_smoother.smooth_bbox = None

        for face in faces:
            raw_bbox = face.bbox.astype(int)
            bbox     = ui_smoother.update(raw_bbox) if len(faces) == 1 else raw_bbox
            x1, y1, x2, y2 = bbox

            if hasattr(face, 'kps') and face.kps is not None and len(face.kps) >= 2:
                eye_dist = np.linalg.norm(face.kps[0] - face.kps[1])
            else:
                eye_dist = 0

            # ---------------------------------------------------
            # 1. FAISS Biometric Recognition (EXECUTES FIRST)
            # ---------------------------------------------------
            embedding = face.embedding.reshape(1, -1)
            faiss.normalize_L2(embedding)
            distances, indices = index.search(embedding, k=1)
            confidence    = float(distances[0][0])
            detected_name = user_mapping.get(indices[0][0], "Unknown")

            # Initialize states for new identities

            if detected_name not in smoothed_ear_state: smoothed_ear_state[detected_name] = 1.0
            if detected_name not in blink_state: blink_state[detected_name] = False

            # ---------------------------------------------------
            # 2. Temporal Liveness (Blink)
            # ---------------------------------------------------
            if hasattr(face, 'landmark_3d_68') and face.landmark_3d_68 is not None:
                raw_ear = calculate_ear_native(face.landmark_3d_68)
                
                # THE BLINK FIX: We bypass the EMA "shock absorber" entirely.
                # We use the raw_ear directly against a slightly more sensitive threshold (0.30).
                # This guarantees instant triggering even if the blink only lasts 1 frame.
                if raw_ear < 0.30:
                    blink_state[detected_name] = True

            # ---------------------------------------------------
            # 3. Spatial Anti-Spoofing (FAS)
            # ---------------------------------------------------
            is_real_face   = True
            liveness_score = 1.0
            
            if fas_session is not None:
                h_img, w_img = frame_to_process.shape[:2]
                # 2.7x Contextual Bounding Box
                face_width, face_height = x2 - x1, y2 - y1
                center_x, center_y = x1 + face_width // 2, y1 + face_height // 2
                new_size = int(max(face_width, face_height) * 2.7)
                
                nx1, ny1 = max(0, center_x - new_size // 2), max(0, center_y - new_size // 2)
                nx2, ny2 = min(w_img, center_x + new_size // 2), min(h_img, center_y + new_size // 2)
                
                face_roi = frame_to_process[ny1:ny2, nx1:nx2]
                
                if face_roi.size > 0:
                    try:
                        fas_input = preprocess_fas_image(face_roi)
                        preds = fas_session.run(None, {fas_session.get_inputs()[0].name: fas_input})[0]
                        raw_liveness = float(stable_softmax(preds[0])[1])
                        
                        if detected_name not in smoothed_liveness_state:
                            smoothed_liveness_state[detected_name] = raw_liveness
                        else:
                            smoothed_liveness_state[detected_name] = (EMA_FAS_ALPHA * raw_liveness + (1 - EMA_FAS_ALPHA) * smoothed_liveness_state[detected_name])
                        
                        liveness_score = smoothed_liveness_state[detected_name]
                        if np.isnan(liveness_score) or liveness_score < FAS_REAL_THRESHOLD:
                            is_real_face = False
                            
                    except Exception: pass

            # ---------------------------------------------------
            # 4. Dynamic Thresholding & Decision Matrix
            # ---------------------------------------------------
            dynamic_threshold = BASE_RECOGNITION_THRESHOLD
            if liveness_score > 0.90: dynamic_threshold -= 0.05
            elif liveness_score < 0.70: dynamic_threshold += 0.05

            if not is_real_face:
                color, status = (0, 165, 255), f"SPOOF DETECTED ({int(liveness_score * 100)}%)"
                log_to_sqlite("IMPOSTER", "SPOOF_ATTEMPT", liveness_score, confidence)
            elif eye_dist < EYE_DIST_MIN:
                color, status = (50, 50, 255), "MOVE CLOSER"
            elif eye_dist > EYE_DIST_MAX:
                color, status = (50, 50, 255), "STEP BACK"
            elif confidence > dynamic_threshold:
                if not blink_state[detected_name]:
                    color, status = (255, 255, 0), f"BLINK TO VERIFY | {detected_name.upper()}"
                else:
                    color, status = (0, 255, 100), f"GRANTED | {detected_name.upper()}"
                    log_to_sqlite(detected_name, "ACCESS_GRANTED", liveness_score, confidence)
                    blink_state[detected_name] = False
                    # ── Signal ESP32 to open gate ────────────────────────────
                    threading.Thread(target=signal_esp32, args=(ESP32_GRANT_URL,), daemon=True).start()
            else:
                color, status = (50, 50, 255), "ACCESS DENIED"
                blink_state[detected_name] = False
                log_to_sqlite("STRANGER", "ACCESS_DENIED", liveness_score, confidence)
                threading.Thread(target=signal_esp32, args=(ESP32_DENY_URL,), daemon=True).start()

            new_detections.append({'bbox': bbox, 'status': status, 'color': color, 'confidence': confidence, 'liveness': liveness_score})

        while not result_queue.empty():
            try: result_queue.get_nowait()
            except Exception: pass
        result_queue.put(new_detections)

    # Cleanup (tickets_conn is the correct variable name)

# ==========================================
# 3. MAIN EXECUTION GUARD
# ==========================================
if __name__ == "__main__":
    mp.freeze_support()
    frame_queue = mp.Queue(maxsize=2)
    result_queue = mp.Queue(maxsize=2)

    worker_process = mp.Process(target=ai_worker, args=(frame_queue, result_queue), daemon=True)
    worker_process.start()
    print("[INFO] Sentinel V2: AI worker process spawned.")

    print(f"[INFO] Connecting to ESP32-CAM stream: {ESP32_STREAM_URL}")
    cap = CameraStream(ESP32_STREAM_URL)
    print("[INFO] Pipeline ready. Press 'q' to quit.")

    detections_to_draw = []

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        frame = cv2.resize(frame, (640, 480))

        if frame_queue.full():
            try: frame_queue.get_nowait()
            except Exception: pass
        frame_queue.put(frame.copy())

        if not result_queue.empty():
            try: detections_to_draw = result_queue.get_nowait()
            except Exception: pass

        for det in detections_to_draw:
            x1, y1, x2, y2 = det['bbox']
            color, status = det['color'], det['status']

            length, thickness = 25, 2
            cv2.line(frame, (x1, y1), (x1 + length, y1), color, thickness)
            cv2.line(frame, (x1, y1), (x1, y1 + length), color, thickness)
            cv2.line(frame, (x2, y1), (x2 - length, y1), color, thickness)
            cv2.line(frame, (x2, y1), (x2, y1 + length), color, thickness)
            cv2.line(frame, (x1, y2), (x1 + length, y2), color, thickness)
            cv2.line(frame, (x1, y2), (x1, y2 - length), color, thickness)
            cv2.line(frame, (x2, y2), (x2 - length, y2), color, thickness)
            cv2.line(frame, (x2, y2), (x2, y2 - length), color, thickness)

            text_color = (0, 255, 0) if color == (0, 255, 100) else (255, 255, 255)
            cv2.putText(frame, status, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(frame, status, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color, 1)

            conf_text = f"Match: {det['confidence']*100:.1f}%  Live: {det['liveness']*100:.1f}%"
            cv2.putText(frame, conf_text, (x1, y1 + 15), cv2.FONT_HERSHEY_DUPLEX, 0.38, (0, 0, 0), 2)
            cv2.putText(frame, conf_text, (x1, y1 + 15), cv2.FONT_HERSHEY_DUPLEX, 0.38, text_color, 1)

        cv2.imshow("SENTINEL V2 PRO", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    frame_queue.put(None)
    worker_process.join(timeout=2)
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Systeme ferme proprement.")
# frontend/ — Project Sentinel V2

> Visitor-facing web portal for FIFA 2030 biometric pass registration.

---

## What This Folder Does

This folder contains the **browser-side layer** of Project Sentinel V2. It is
the only part of the system that a visitor ever interacts with directly.

```
Visitor opens browser
  └─▶  sentinel-biometric.html   (5-step registration wizard)
         │
         ├─▶  POST /enroll  ──▶  software/server.py  ──▶  1_enroll_users.py
         │                        (FastAPI on port 8000)       (InsightFace + FAISS)
         │
         └─▶  saveTicketToFirebase()  ──▶  firebase.js  ──▶  Firestore
```

The flow in plain English:
1. Visitor fills in passport / travel details (Step 1).
2. Webcam opens, liveness is checked client-side (Step 2).
3. On liveness success, a JPEG frame is captured to a hidden `<canvas>` and
   sent as a Base64 data-URL to `POST /enroll`.
4. The FastAPI server decodes the image, calls InsightFace, stores the 512D
   vector in FAISS, and returns a `secure_id` + vector preview.
5. Visitor chooses stand zone & ticket category (Step 3).
6. On final confirmation (Step 4), `saveTicketToFirebase()` writes the full
   ticket record (passenger + biometric metadata) to Firestore.
7. At the physical gate, the visitor's face is matched against the FAISS index
   by `2_live_gate.py` — **the face is the ticket**.

---

## Files in This Folder

| File | Purpose |
|---|---|
| `sentinel-biometric.html` | Single-page app — all HTML, CSS, and JavaScript in one file. Drives the 5-step registration flow. |
| `firebase.js` | ES module. Initialises the Firebase app and exports `saveTicketToFirebase()`. Imported by the HTML via `<script type="module">`. |
| `README.md` | This file. |

---

## Serving the Frontend Locally

The HTML file uses ES module imports (`type="module"`) and calls a local API,
so it **must** be served over HTTP — do not open it directly as a `file://` URL
(the browser will block module imports and camera access).

### Option A — Python (no install needed)

```bash
# From the project root
cd Y:/Sentinel-original/Project-Sentinel/frontend
python -m http.server 5500
# Open: http://localhost:5500/sentinel-biometric.html
```

### Option B — VS Code Live Server

Install the **Live Server** extension, right-click `sentinel-biometric.html`
→ **Open with Live Server**.  Make sure Live Server serves on port 5500 (its
default).

### Option C — Node http-server

```bash
npx http-server frontend/ -p 5500 --cors
# Open: http://localhost:5500/sentinel-biometric.html
```

---

## Connecting to `software/server.py`

### Starting the API

```bash
# From the project root (Y:/Sentinel-original/Project-Sentinel/)
uvicorn software.server:app --reload --port 8000
```

The server must be running **before** a visitor reaches Step 2 (face scan).

### API Base URL

The frontend uses a single constant at the top of the `<script type="module">`
block inside `sentinel-biometric.html`:

```js
const API_BASE = 'http://127.0.0.1:8000';
```

**Do not change the format of this URL** (no trailing slash, no path prefix).
All fetch calls are constructed as `` `${API_BASE}/enroll` ``.

### `/enroll` Endpoint

| Property | Value |
|---|---|
| Method | `POST` |
| URL | `http://127.0.0.1:8000/enroll` |
| Content-Type | `application/json` |
| Auth | None (add in production — see Improvement suggestions) |

**Request body** (all fields required):

```json
{
  "name":        "JOHN DOE",
  "passport":    "AB1234567",
  "email":       "john@example.com",
  "nationality": "Morocco",
  "gender":      "Male",
  "dob":         "1990-06-15",
  "match":       "FIFA 2030 Final — Grand Stade, Casablanca",
  "zone":        "North",
  "category":    "VIP",
  "image_data":  "data:image/jpeg;base64,/9j/4AAQ..."
}
```

**Success response** (`200 OK`):

```json
{
  "success":        true,
  "secure_id":      "SEN-0042-A3F7",
  "vector_dim":     512,
  "vector_preview": [0.1234, -0.5678, ...],
  "faiss_row":      42,
  "enrolled_at":    "2026-04-25T02:00:00Z",
  "message":        "Enrolled OK"
}
```

**Error response** (`422 Unprocessable Entity`):

```json
{
  "success": false,
  "detail":  "No face detected in image."
}
```

### Health Check

```bash
curl http://127.0.0.1:8000/health
# {"status":"online","service":"Sentinel Biometric API v2"}
```

### Interactive Docs

FastAPI auto-generates Swagger UI at `http://127.0.0.1:8000/docs`.

---

## How `firebase.js` Connects to Firestore

`firebase.js` uses the **Firebase JS SDK v10** loaded directly from the
Google CDN (no `npm install` required).  It exports a single function:

```js
saveTicketToFirebase(ticketData) → Promise<{ success, docId, error }>
```

This is called by `sentinel-biometric.html` at the final confirmation step.
It writes one document to the Firestore collection **`sentinel_registrations`**.

### Document Schema

```js
{
  secureId : "SEN-0042-A3F7",
  passenger: { name, passport, email, nationality, gender, dob },
  ticket   : { match, zone, category },
  biometric: { type, engine, verified, vectorDim, vectorPreview[], enrolledAt },
  createdAt: <Firestore server timestamp>
}
```

### Firebase Project Config

The config values are currently hard-coded in `firebase.js` (lines 16-24).
The relevant values a developer may need to update:

| Key | Description |
|---|---|
| `apiKey` | Firebase Web API key (public) |
| `authDomain` | `<project-id>.firebaseapp.com` |
| `projectId` | Firestore project ID |
| `storageBucket` | Firebase Storage bucket (not used for ticket data) |
| `messagingSenderId` | Cloud Messaging sender ID |
| `appId` | Firebase app ID |
| `measurementId` | Google Analytics measurement ID |

To use a **different Firebase project**, replace the entire `firebaseConfig`
object in `firebase.js` with the config from your Firebase Console:
**Project Settings → General → Your apps → SDK setup and configuration**.

Firestore Security Rules must allow writes from the browser.  Minimum rule for
development:

```
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /sentinel_registrations/{doc} {
      allow write: if true;   // ⚠ Tighten before production
    }
  }
}
```

---

## Environment Variables / Config Values a Developer Must Set

| What | Where | Default / Note |
|---|---|---|
| API server host & port | `sentinel-biometric.html` — `const API_BASE` | `http://127.0.0.1:8000` |
| Firebase project config | `firebase.js` — `firebaseConfig` object | Set to `sentinel-fef8b` |
| Firestore collection name | `firebase.js` — `collection(_db, ...)` | `sentinel_registrations` |
| CORS origins on server | `software/server.py` — `allow_origins` | `["*"]` — restrict in production |

---

## What NOT to Change

The items below are load-bearing integration points. Changing them without
coordinating with the backend will silently break enrollment.

| Item | File | Why it must not change |
|---|---|---|
| `const API_BASE = 'http://127.0.0.1:8000'` format | `sentinel-biometric.html` | All fetch calls append `/enroll` to this string directly |
| Canvas mirror logic (`ctx.translate` + `ctx.scale(-1,1)`) | `sentinel-biometric.html` — `captureVector()` | The CSS preview is mirrored; the canvas un-mirrors before sending so the server receives a natural-orientation image. Removing this causes InsightFace to receive a horizontally flipped face |
| `toDataURL('image/jpeg', 0.92)` | `sentinel-biometric.html` — `captureVector()` | `server.py`'s `_decode_image()` expects a JPEG data-URL |
| `image_data` key in the POST body | `sentinel-biometric.html` | Matched by `EnrollRequest.image_data` in `server.py` |
| `saveTicketToFirebase` export name | `firebase.js` | Imported by exact name in the `<script type="module">` block |
| Firestore collection name `sentinel_registrations` | `firebase.js` | Gate-side queries and admin dashboards reference this collection |
| InsightFace model name `buffalo_s`, `det_size`, FAISS index type | `software/1_enroll_users.py` | **Out of scope for this folder entirely — do not suggest changes** |

---

## Known Gotcha — Python Module Import in `server.py`

`1_enroll_users.py` starts with a digit, which Python's `import` statement
cannot handle.  `server.py` works around this with `importlib.util`:

```python
_mod_path = pathlib.Path(__file__).parent / "1_enroll_users.py"
_spec     = importlib.util.spec_from_file_location("enroll_engine", _mod_path)
```

This is intentional and must not be "simplified" by renaming `1_enroll_users.py`.

---

*Project Sentinel V2 — FIFA 2030 Biometric Access · Security Unit*

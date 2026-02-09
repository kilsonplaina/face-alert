import os
import time
import sqlite3
import pickle
import threading
from datetime import datetime
from functools import wraps
import secrets
import hashlib

import cv2
import numpy as np
from flask import (
    Flask, render_template, request, redirect, session,
    jsonify, Response, flash
)

import face_recognition

# Alarme sonoro (Windows)
try:
    import winsound
    _HAS_WINSOUND = True
except Exception:
    _HAS_WINSOUND = False

# =====================================================
# CONFIG
# =====================================================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STATIC = os.path.join(BASE_DIR, "static")
UPLOAD = os.path.join(STATIC, "uploads")
SNAPS = os.path.join(STATIC, "snapshots")
CAPS = os.path.join(STATIC, "captures")
DB_PATH = os.path.join(BASE_DIR, "facealert.db")

for p in (UPLOAD, SNAPS, CAPS):
    os.makedirs(p, exist_ok=True)

CAM_INDEX = 0
FRAME_W = 1280
FRAME_H = 720

# === DISTINÇÃO ===
THRESHOLD = 0.45
MIN_FACE_SIZE = 90
EVENT_COOLDOWN_SEC = 8

# === ALARME SONORO (CASA: só desconhecidos) ===
ALARM_ENABLED = True
ALARM_ONLY_UNKNOWN = True
ALARM_COOLDOWN_SEC = 15       # toca no máximo 1x a cada 15s
ALARM_BEEPS = 3
ALARM_FREQ = 1800
ALARM_DUR_MS = 250

# =====================================================
# APP
# =====================================================
app = Flask(__name__)
app.secret_key = os.environ.get("FACEALERT_SECRET", secrets.token_hex(32))

# =====================================================
# CAMERA (abre sob demanda + lock)
# =====================================================
_camera = None
camera_lock = threading.Lock()

def get_camera():
    global _camera

    with camera_lock:
        if _camera is not None:
            try:
                if _camera.isOpened():
                    return _camera
            except Exception:
                pass
            _camera = None

        cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(CAM_INDEX)

        if cap is None or not cap.isOpened():
            return None

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

        _camera = cap
        return _camera

# =====================================================
# DATABASE
# =====================================================
def db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = db()
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'admin',
        created_at TEXT NOT NULL
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS people(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        image TEXT NOT NULL,
        encoding BLOB NOT NULL,
        created_at TEXT NOT NULL
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS events(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        person TEXT NOT NULL,
        confidence REAL NOT NULL,
        snapshot TEXT NOT NULL
    )
    """)

    conn.commit()
    conn.close()

# =====================================================
# AUTH / SECURITY
# =====================================================
def now_iso():
    return datetime.now().isoformat(timespec="seconds")

def hash_password(pw: str) -> str:
    return hashlib.sha256((pw or "").encode("utf-8")).hexdigest()

def ensure_admin():
    conn = db()
    row = conn.execute("SELECT id FROM users WHERE username=?", ("admin",)).fetchone()
    if not row:
        conn.execute(
            "INSERT INTO users (username, password_hash, role, created_at) VALUES (?,?,?,?)",
            ("admin", hash_password("admin123"), "admin", now_iso())
        )
        conn.commit()
    conn.close()

def logged():
    return bool(session.get("uid"))

def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not logged():
            return redirect("/")
        return fn(*args, **kwargs)
    return wrapper

# =====================================================
# HELPERS
# =====================================================
def safe_name(s: str) -> str:
    s = (s or "").strip()
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("_", "-"):
            out.append(ch)
        elif ch == " ":
            out.append("_")
    r = "".join(out)
    return r[:50] if r else "person"

def abs_from_rel(rel_path: str) -> str:
    rel = rel_path.lstrip("/").replace("/", os.sep)
    return os.path.join(BASE_DIR, rel)

# =====================================================
# ALARME SONORO (thread)
# =====================================================
def play_alarm():
    if not ALARM_ENABLED:
        return

    def _beep():
        try:
            if _HAS_WINSOUND:
                for _ in range(ALARM_BEEPS):
                    winsound.Beep(ALARM_FREQ, ALARM_DUR_MS)
                    time.sleep(0.05)
            else:
                for _ in range(ALARM_BEEPS):
                    print("\a", end="", flush=True)
                    time.sleep(0.12)
        except Exception:
            pass

    threading.Thread(target=_beep, daemon=True).start()

# Estado do alarme inteligente (só desconhecido novo)
_last_unknown_seen = 0.0
_unknown_active_until = 0.0

# =====================================================
# CACHE DE PESSOAS (não carrega no DB a cada frame)
# =====================================================
known_lock = threading.Lock()
known_names = []
known_encs = []
known_updated_at = 0.0

def refresh_known_people():
    global known_names, known_encs, known_updated_at

    conn = db()
    rows = conn.execute("SELECT name, encoding FROM people").fetchall()
    conn.close()

    names = []
    encs = []
    for r in rows:
        try:
            names.append(r["name"])
            encs.append(pickle.loads(r["encoding"]))
        except Exception:
            pass

    with known_lock:
        known_names = names
        known_encs = encs
        known_updated_at = time.time()

def get_known_people():
    with known_lock:
        return list(known_names), list(known_encs)

# =====================================================
# RECONHECIMENTO (best match + threshold)
# =====================================================
def recognize_face(face_encoding, names, encs):
    if not encs:
        return ("DESCONHECIDO", 0.0, 1.0)

    dists = face_recognition.face_distance(encs, face_encoding)
    best_idx = int(np.argmin(dists))
    best_dist = float(dists[best_idx])

    if best_dist < THRESHOLD:
        name = names[best_idx]
        confidence = max(0.0, min(1.0, 1.0 - best_dist))
        return (name, confidence, best_dist)

    return ("DESCONHECIDO", 0.0, best_dist)

# =====================================================
# EVENT LOGIC (cooldown por pessoa)
# =====================================================
last_event = {}

def can_log_event(person: str):
    t = time.time()
    last = last_event.get(person, 0.0)
    if (t - last) >= EVENT_COOLDOWN_SEC:
        last_event[person] = t
        return True
    return False

def save_event(person: str, confidence: float, frame_bgr):
    fn = f"event_{int(time.time())}_{safe_name(person)}.jpg"
    abs_path = os.path.join(SNAPS, fn)
    rel_path = f"/static/snapshots/{fn}"

    try:
        cv2.imwrite(abs_path, frame_bgr)
    except Exception:
        rel_path = "/static/snapshots/"

    conn = db()
    conn.execute(
        "INSERT INTO events (ts, person, confidence, snapshot) VALUES (?,?,?,?)",
        (now_iso(), person, float(confidence), rel_path),
    )
    conn.commit()
    conn.close()

# =====================================================
# ROUTES
# =====================================================
@app.route("/", methods=["GET", "POST"])
def login():
    # configurações do bloqueio
    MAX_TRIES = 5
    LOCK_MINUTES = 15

    # se estiver bloqueado
    lock_until = session.get("lock_until", 0)
    if lock_until and time.time() < lock_until:
        remain = int((lock_until - time.time()) / 60) + 1
        flash(f"Muitas tentativas. Tente novamente em ~{remain} min.", "error")
        return render_template("login.html")

    if request.method == "POST":
        user = (request.form.get("user") or "").strip()
        pw = request.form.get("pass") or ""

        conn = db()
        row = conn.execute(
            "SELECT id, username, role, password_hash FROM users WHERE username=?",
            (user,)
        ).fetchone()
        conn.close()

        if row and row["password_hash"] == hash_password(pw):
            # login OK: zera tentativas e cria sessão
            session.clear()
            session["uid"] = row["id"]
            session["user"] = row["username"]
            session["role"] = row["role"]
            return redirect("/dashboard")

        # login falhou: conta tentativa
        tries = int(session.get("tries", 0)) + 1
        session["tries"] = tries

        if tries >= MAX_TRIES:
            session["lock_until"] = time.time() + (LOCK_MINUTES * 60)
            session["tries"] = 0
            flash(f"Muitas tentativas. Bloqueado por {LOCK_MINUTES} minutos.", "error")
        else:
            left = MAX_TRIES - tries
            flash(f"Usuário ou senha inválidos. Restam {left} tentativas.", "error")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

@app.route("/dashboard")
@login_required
def dashboard():
    conn = db()
    people_count = conn.execute("SELECT COUNT(*) AS c FROM people").fetchone()["c"]
    events_count = conn.execute("SELECT COUNT(*) AS c FROM events").fetchone()["c"]

    recent_rows = conn.execute(
        "SELECT ts, person, confidence, snapshot FROM events ORDER BY id DESC LIMIT 5"
    ).fetchall()
    recent = [dict(r) for r in recent_rows]

    conn.close()

    return render_template(
        "dashboard.html",
        people_count=people_count,
        events_count=events_count,
        recent=recent
    )

@app.route("/monitor")
@login_required
def monitor():
    return render_template("monitor.html")

@app.route("/people")
@login_required
def people():
    conn = db()
    rows = conn.execute("SELECT * FROM people ORDER BY id DESC").fetchall()
    conn.close()
    return render_template("people.html", people=rows)

# =====================================================
# CAPTURE (tirar foto da câmera)
# =====================================================
@app.post("/api/capture")
def api_capture():
    if not logged():
        return jsonify({"ok": False, "error": "not_logged"}), 401

    cam = get_camera()
    if cam is None:
        return jsonify({"ok": False, "error": "camera_not_open"}), 500

    frame = None
    for _ in range(10):
        with camera_lock:
            ok, f = cam.read()
        if ok and f is not None:
            frame = f
            break
        time.sleep(0.05)

    if frame is None:
        return jsonify({"ok": False, "error": "camera_read_failed"}), 500

    fn = f"cap_{int(time.time())}.jpg"
    abs_path = os.path.join(CAPS, fn)
    rel_path = f"/static/captures/{fn}"
    cv2.imwrite(abs_path, frame)

    return jsonify({"ok": True, "path": rel_path})

# =====================================================
# PEOPLE ADD (upload OU captura)
# =====================================================
@app.post("/people/add")
def people_add():
    if not logged():
        return redirect("/")

    name = (request.form.get("name") or "").strip()
    captured_path = (request.form.get("captured_path") or "").strip()
    photo = request.files.get("photo")

    if not name:
        flash("Digite o nome da pessoa.", "error")
        return redirect("/people")

    abs_path = None
    rel_path = None

    try:
        if captured_path:
            rel_path = captured_path
            abs_path = abs_from_rel(captured_path)
            if not os.path.exists(abs_path):
                flash(f"ERRO: captura não encontrada: {captured_path}", "error")
                return redirect("/people")
        else:
            if not photo or photo.filename.strip() == "":
                flash("Selecione uma foto OU clique em Tirar Foto.", "error")
                return redirect("/people")

            ext = os.path.splitext(photo.filename)[1].lower()
            if ext not in [".jpg", ".jpeg", ".png"]:
                ext = ".jpg"

            filename = f"{int(time.time())}_{safe_name(name)}{ext}"
            abs_path = os.path.join(UPLOAD, filename)
            rel_path = f"/static/uploads/{filename}"
            photo.save(abs_path)

        if not abs_path or not os.path.exists(abs_path):
            flash("ERRO: arquivo não foi salvo/encontrado no disco.", "error")
            return redirect("/people")

        img = face_recognition.load_image_file(abs_path)
        locs = face_recognition.face_locations(img, model="hog", number_of_times_to_upsample=2)

        if len(locs) != 1:
            if rel_path.startswith("/static/uploads/"):
                try:
                    os.remove(abs_path)
                except Exception:
                    pass
            flash(f"Foto inválida: foram detectados {len(locs)} rostos. Use APENAS 1 rosto.", "error")
            return redirect("/people")

        enc = face_recognition.face_encodings(img, known_face_locations=locs)[0]
        enc_blob = sqlite3.Binary(pickle.dumps(enc))

        conn = db()
        try:
            conn.execute(
                "INSERT INTO people (name, image, encoding, created_at) VALUES (?,?,?,?)",
                (name, rel_path, enc_blob, now_iso()),
            )
            conn.commit()
        finally:
            conn.close()

        refresh_known_people()
        flash(f"✅ Pessoa cadastrada: {name}", "ok")
        return redirect("/people")

    except Exception as e:
        print("ERRO people_add:", repr(e))
        flash("ERRO AO CADASTRAR (detalhe): " + repr(e), "error")
        return redirect("/people")

# =====================================================
# PEOPLE DELETE
# =====================================================
@app.post("/people/delete")
def people_delete():
    if not logged():
        return redirect("/")

    pid = int(request.form.get("id", "0"))
    conn = db()
    row = conn.execute("SELECT image FROM people WHERE id=?", (pid,)).fetchone()

    if row:
        try:
            abs_path = abs_from_rel(row["image"])
            if os.path.exists(abs_path) and row["image"].startswith("/static/uploads/"):
                os.remove(abs_path)
        except Exception:
            pass

    conn.execute("DELETE FROM people WHERE id=?", (pid,))
    conn.commit()
    conn.close()

    refresh_known_people()
    flash("Pessoa removida.", "ok")
    return redirect("/people")

# =====================================================
# VIDEO STREAM (distinção + filtro + eventos + alarme inteligente)
# =====================================================
def gen_frames():
    global _last_unknown_seen, _unknown_active_until

    if known_updated_at == 0:
        refresh_known_people()

    while True:
        cam = get_camera()
        if cam is None:
            img = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(img, "ERRO: Camera nao abriu (feche apps da webcam)", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            ok, buf = cv2.imencode(".jpg", img)
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            time.sleep(1)
            continue

        with camera_lock:
            success, frame = cam.read()

        if not success or frame is None:
            time.sleep(0.05)
            continue

        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        face_locs = face_recognition.face_locations(rgb, model="hog")
        face_encs = face_recognition.face_encodings(rgb, face_locs)

        names, encs = get_known_people()

        for (t, rgt, b, l), enc in zip(face_locs, face_encs):
            t2, r2, b2, l2 = t*2, rgt*2, b*2, l*2

            w = r2 - l2
            h = b2 - t2
            if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                continue

            person, conf, dist = recognize_face(enc, names, encs)

            # === ALARME INTELIGENTE: só desconhecido "novo" ===
            now = time.time()
            if person == "DESCONHECIDO" and ALARM_ENABLED and ALARM_ONLY_UNKNOWN:
                _last_unknown_seen = now
                if now > _unknown_active_until:
                    play_alarm()
                    _unknown_active_until = now + ALARM_COOLDOWN_SEC

            color = (0, 255, 0) if person != "DESCONHECIDO" else (0, 0, 255)

            cv2.rectangle(frame, (l2, t2), (r2, b2), color, 2)
            cv2.putText(frame, f"{person}  conf:{conf:.2f}  d:{dist:.2f}",
                        (l2, max(25, t2 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # eventos continuam a ser registados (como tinhas)
            if can_log_event(person):
                try:
                    save_event(person, conf, frame)
                except Exception:
                    pass

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

@app.route("/video_feed")
@login_required
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# =====================================================
# API
# =====================================================
@app.get("/api/people")
def api_people():
    if not logged():
        return jsonify({"items": []})

    conn = db()
    rows = conn.execute("SELECT id,name,image,created_at FROM people ORDER BY id DESC").fetchall()
    conn.close()
    return jsonify({"items": [dict(r) for r in rows]})

@app.get("/api/settings")
def api_settings():
    return jsonify({
        "threshold": THRESHOLD,
        "min_face_size": MIN_FACE_SIZE,
        "event_cooldown_sec": EVENT_COOLDOWN_SEC,
        "alarm_enabled": ALARM_ENABLED,
        "alarm_only_unknown": ALARM_ONLY_UNKNOWN,
        "alarm_cooldown_sec": ALARM_COOLDOWN_SEC,
        "alarm_beeps": ALARM_BEEPS,
        "alarm_freq": ALARM_FREQ,
        "alarm_dur_ms": ALARM_DUR_MS
    })

# =====================================================
if __name__ == "__main__":
    init_db()
    ensure_admin()
    refresh_known_people()
    print("FaceAlert ativo em http://127.0.0.1:8010")
    app.run(host="127.0.0.1", port=8010, debug=True, threaded=True)

import io
import json
import math
import os
import re
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, flash, redirect, render_template, request, url_for
from PIL import Image

try:
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover
    genai = None
    types = None

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRAINING_DIR = DATA_DIR / "training"
DB_PATH = DATA_DIR / "yarn_detector.db"

DATA_DIR.mkdir(exist_ok=True)
TRAINING_DIR.mkdir(exist_ok=True)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS yarn_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT NOT NULL,
            rgb_r INTEGER NOT NULL,
            rgb_g INTEGER NOT NULL,
            rgb_b INTEGER NOT NULL,
            hex TEXT NOT NULL,
            source TEXT,
            created_at TEXT NOT NULL,
            gemini_summary TEXT
        )
        """
    )
    conn.commit()
    conn.close()


init_db()


def add_yarn_sample(label, rgb_tuple, source, gemini_summary):
    r, g, b = rgb_tuple
    created_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT INTO yarn_samples (label, rgb_r, rgb_g, rgb_b, hex, source, created_at, gemini_summary)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (label, r, g, b, rgb_to_hex((r, g, b)), source, created_at, gemini_summary),
    )
    conn.commit()
    conn.close()
    return {
        "label": label,
        "rgb": [r, g, b],
        "hex": rgb_to_hex((r, g, b)),
        "source": source,
        "created_at": created_at,
        "gemini_summary": gemini_summary,
    }


def load_palette():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT label, rgb_r, rgb_g, rgb_b, hex, source, created_at, gemini_summary
        FROM yarn_samples
        ORDER BY created_at DESC
        """
    ).fetchall()
    conn.close()

    palette = []
    for row in rows:
        palette.append({
            "label": row["label"],
            "rgb": [row["rgb_r"], row["rgb_g"], row["rgb_b"]],
            "hex": row["hex"],
            "source": row["source"],
            "created_at": row["created_at"],
            "gemini_summary": row["gemini_summary"],
        })
    return palette


def save_palette(entries):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM yarn_samples")
    for item in entries:
        rgb = item.get("rgb", [0, 0, 0])
        conn.execute(
            """
            INSERT INTO yarn_samples (label, rgb_r, rgb_g, rgb_b, hex, source, created_at, gemini_summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item.get("label", "Unknown"),
                int(rgb[0]),
                int(rgb[1]),
                int(rgb[2]),
                item.get("hex", rgb_to_hex(tuple(rgb))),
                item.get("source", "manual"),
                item.get("created_at", datetime.utcnow().isoformat(timespec="seconds") + "Z"),
                item.get("gemini_summary", "unknown"),
            ),
        )
    conn.commit()
    conn.close()


def rgb_to_hex(rgb):
    return "#%02x%02x%02x" % tuple(max(0, min(255, int(v))) for v in rgb)


def color_distance(rgb_a, rgb_b):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(rgb_a, rgb_b)))


def normalize_label(label):
    text = re.sub(r"\s+", " ", label or "").strip()
    return text.title() if text else "Unknown"


def detect_dominant_rgb(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((160, 160))
    pixels = list(image.getdata())

    buckets = {}
    for r, g, b in pixels:
        key = (r // 32, g // 32, b // 32)
        buckets.setdefault(key, []).append((r, g, b))

    if not buckets:
        return (0, 0, 0)

    dominant_bucket = max(buckets.items(), key=lambda item: len(item[1]))[1]
    avg_r = sum(p[0] for p in dominant_bucket) // len(dominant_bucket)
    avg_g = sum(p[1] for p in dominant_bucket) // len(dominant_bucket)
    avg_b = sum(p[2] for p in dominant_bucket) // len(dominant_bucket)
    return (avg_r, avg_g, avg_b)


def infer_best_match(current_rgb, palette):
    if not palette:
        return None

    scored = []
    for item in palette:
        item_rgb = tuple(item.get("rgb", (0, 0, 0)))
        score = color_distance(current_rgb, item_rgb)
        scored.append({
            "label": item.get("label", "Unknown"),
            "hex": item.get("hex", rgb_to_hex(item_rgb)),
            "rgb": item_rgb,
            "distance": score,
            "source": item.get("source", "manual"),
        })

    return min(scored, key=lambda item: item["distance"])


def guess_mime_type(filename):
    lower = (filename or "").lower()
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith(".webp"):
        return "image/webp"
    return "image/jpeg"


def parse_gemini_response(raw_output):
    cleaned = (raw_output or "").strip()
    if not cleaned:
        return {"dominant_color": "unknown", "confidence": 0, "notes": "No response from Gemini."}

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        payload = json.loads(cleaned)
        return {
            "dominant_color": payload.get("dominant_color", "unknown"),
            "confidence": payload.get("confidence", 0),
            "notes": payload.get("notes", "Gemini analyzed the yarn."),
        }
    except json.JSONDecodeError:
        return {"dominant_color": "unknown", "confidence": 0, "notes": cleaned[:180]}


def create_prediction_result(image_bytes, filename="sample.jpg"):
    dominant_rgb = detect_dominant_rgb(image_bytes)
    mime_type = guess_mime_type(filename)
    gemini = analyze_with_gemini(image_bytes, mime_type)
    palette = load_palette()
    match = infer_best_match(dominant_rgb, palette)
    score = 100 - min(99, int((match["distance"] if match else 400) / 4)) if match else 0

    return {
        "current_hex": rgb_to_hex(dominant_rgb),
        "current_rgb": list(dominant_rgb),
        "matched_label": match["label"] if match else "No saved yarn yet",
        "matched_hex": match["hex"] if match else "#000000",
        "match_score": max(0, score),
        "distance": round(match["distance"], 2) if match else None,
        "gemini": gemini,
    }


def analyze_with_gemini(image_bytes, mime_type):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or genai is None:
        return {
            "dominant_color": "unknown",
            "confidence": 0,
            "notes": "Gemini API is not configured. Add GEMINI_API_KEY to your .env file.",
            "enabled": False,
        }

    try:
        client = genai.Client(api_key=api_key)
        prompt = (
            "Look at this yarn photo. Identify the dominant yarn color and return JSON only with "
            "keys: dominant_color, confidence, notes. Keep the color name simple like red, blue, "
            "green, black, white, gray, yellow, orange, purple, pink, brown, beige, teal, navy, olive, maroon."
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                prompt,
            ],
        )
        return {**parse_gemini_response(getattr(response, "text", "")), "enabled": True}
    except Exception as exc:  # pragma: no cover
        return {
            "dominant_color": "unknown",
            "confidence": 0,
            "notes": f"Gemini analysis failed: {exc}",
            "enabled": True,
        }


@app.route("/")
def index():
    palette = load_palette()
    return render_template("index.html", palette=palette, result=None)


@app.route("/train", methods=["POST"])
def train():
    image_file = request.files.get("image")
    label = normalize_label(request.form.get("label", ""))

    if not image_file or not image_file.filename:
        flash("Please upload a yarn photo first.")
        return redirect(url_for("index"))

    if not label or label == "Unknown":
        flash("Please enter a yarn color name, such as Red, Blue, or Olive.")
        return redirect(url_for("index"))

    image_bytes = image_file.read()
    dominant_rgb = detect_dominant_rgb(image_bytes)
    mime_type = guess_mime_type(image_file.filename)
    gemini = analyze_with_gemini(image_bytes, mime_type)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", label.lower()).strip("_") or "yarn"
    file_name = f"{safe_name}_{timestamp}_{uuid.uuid4().hex[:6]}"
    saved_path = TRAINING_DIR / f"{file_name}.jpg"
    with saved_path.open("wb") as out_file:
        out_file.write(image_bytes)

    add_yarn_sample(
        label=label,
        rgb_tuple=dominant_rgb,
        source=saved_path.name,
        gemini_summary=gemini.get("dominant_color", "unknown"),
    )

    flash(f"Saved {label} in your yarn library. Dominant RGB: {rgb_to_hex(dominant_rgb)}")
    return redirect(url_for("index"))


@app.route("/predict", methods=["POST"])
def predict():
    image_file = request.files.get("image")
    if not image_file or not image_file.filename:
        flash("Please upload a photo to classify.")
        return redirect(url_for("index"))

    palette = load_palette()
    result = create_prediction_result(image_file.read(), image_file.filename)
    return render_template("index.html", palette=palette, result=result)


@app.route("/api/detect", methods=["POST"])
def api_detect():
    image_file = request.files.get("image")
    if not image_file or not image_file.filename:
        return {"error": "No image uploaded."}, 400

    result = create_prediction_result(image_file.read(), image_file.filename)
    return result


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)

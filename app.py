import io
import os
import re
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, flash, redirect, render_template, request, url_for
from PIL import Image

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
    columns = [row[1] for row in conn.execute("PRAGMA table_info(yarn_samples)").fetchall()]
    if "thickness_px" not in columns:
        conn.execute("DROP TABLE IF EXISTS yarn_samples")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS yarn_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT NOT NULL,
            thickness_px REAL NOT NULL,
            source TEXT,
            created_at TEXT NOT NULL,
            notes TEXT
        )
        """
    )
    conn.commit()
    conn.close()


init_db()


def add_yarn_sample(label, thickness_px, source, notes):
    created_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT INTO yarn_samples (label, thickness_px, source, created_at, notes)
        VALUES (?, ?, ?, ?, ?)
        """,
        (label, float(thickness_px), source, created_at, notes),
    )
    conn.commit()
    conn.close()
    return {
        "label": label,
        "thickness_px": float(thickness_px),
        "source": source,
        "created_at": created_at,
        "notes": notes,
    }


def load_palette():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT label, thickness_px, source, created_at, notes
        FROM yarn_samples
        ORDER BY created_at DESC
        """
    ).fetchall()
    conn.close()

    palette = []
    for row in rows:
        palette.append({
            "label": row["label"],
            "thickness_px": float(row["thickness_px"]),
            "source": row["source"],
            "created_at": row["created_at"],
            "notes": row["notes"],
        })
    return palette


def save_palette(entries):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM yarn_samples")
    for item in entries:
        conn.execute(
            """
            INSERT INTO yarn_samples (label, thickness_px, source, created_at, notes)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                item.get("label", "Unknown"),
                float(item.get("thickness_px", 0.0)),
                item.get("source", "manual"),
                item.get("created_at", datetime.utcnow().isoformat(timespec="seconds") + "Z"),
                item.get("notes", "reference"),
            ),
        )
    conn.commit()
    conn.close()


def normalize_label(label):
    text = re.sub(r"\s+", " ", label or "").strip()
    return text.title() if text else "Reference"


def build_auto_reference_label(thickness_px):
    return f"Reference {round(float(thickness_px), 2)} px"


def guess_mime_type(filename):
    lower = (filename or "").lower()
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith(".webp"):
        return "image/webp"
    return "image/jpeg"


def measure_yarn_thickness(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("L")
    except OSError:
        return 0.0

    base_width = 600
    if image.width > image.height:
        new_width = base_width
        new_height = max(1, int(image.height * (base_width / image.width)))
    else:
        new_height = base_width
        new_width = max(1, int(image.width * (base_width / image.height)))

    image = image.resize((new_width, new_height))
    pixels = image.load()
    dark_points = []

    for y in range(image.height):
        for x in range(image.width):
            if pixels[x, y] < 180:
                dark_points.append((x, y))

    if not dark_points:
        return float(max(image.width, image.height)) / 2.0

    xs = [x for x, _ in dark_points]
    ys = [y for _, y in dark_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max_x - min_x + 1
    height = max_y - min_y + 1

    row_widths = []
    for y in range(min_y, max_y + 1):
        row_pixels = [x for x in range(min_x, max_x + 1) if pixels[x, y] < 180]
        if row_pixels:
            row_widths.append(max(row_pixels) - min(row_pixels) + 1)

    if row_widths:
        return float(sum(row_widths) / len(row_widths))

    return float(max(width, height))


def infer_best_match(current_thickness, palette):
    if not palette:
        return None

    best = None
    for item in palette:
        target = float(item.get("thickness_px", 0.0))
        distance = abs(current_thickness - target)
        if best is None or distance < best["distance"]:
            best = {
                "label": item.get("label", "Unknown"),
                "target": target,
                "distance": distance,
                "source": item.get("source", "manual"),
            }
    return best


def create_prediction_result(image_bytes, filename="sample.jpg"):
    measured_thickness = measure_yarn_thickness(image_bytes)
    palette = load_palette()
    match = infer_best_match(measured_thickness, palette) if palette else None

    if not match:
        return {
            "status": "not acceptable",
            "message": "No saved yarn thickness reference yet.",
            "measured_thickness_px": round(measured_thickness, 2),
            "reference_label": "No sample",
            "reference_thickness_px": None,
            "difference_px": None,
            "match_score": 0,
            "acceptable": False,
        }

    tolerance = max(2.0, match["target"] * 0.15)
    difference = abs(measured_thickness - match["target"])
    acceptable = difference <= tolerance
    score = max(0, min(100, int(100 - (difference / max(match["target"], 1.0)) * 100)))

    return {
        "status": "acceptable" if acceptable else "not acceptable",
        "message": "acceptable" if acceptable else "not acceptable",
        "measured_thickness_px": round(measured_thickness, 2),
        "reference_label": match["label"],
        "reference_thickness_px": round(match["target"], 2),
        "difference_px": round(difference, 2),
        "tolerance_px": round(tolerance, 2),
        "match_score": score,
        "acceptable": acceptable,
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

    image_bytes = image_file.read()
    thickness = measure_yarn_thickness(image_bytes)

    if not label or label == "Reference":
        label = build_auto_reference_label(thickness)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", label.lower()).strip("_") or "yarn"
    file_name = f"{safe_name}_{timestamp}_{uuid.uuid4().hex[:6]}"
    saved_path = TRAINING_DIR / f"{file_name}.jpg"
    with saved_path.open("wb") as out_file:
        out_file.write(image_bytes)

    add_yarn_sample(
        label=label,
        thickness_px=thickness,
        source=saved_path.name,
        notes=f"Measured thickness {round(thickness, 2)} px",
    )

    flash(f"Saved {label}. Measured thickness: {round(thickness, 2)} px")
    return redirect(url_for("index"))


@app.route("/predict", methods=["POST"])
def predict():
    image_file = request.files.get("image")
    if not image_file or not image_file.filename:
        flash("Please upload a yarn photo to compare.")
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

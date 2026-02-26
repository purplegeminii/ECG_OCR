from flask import Flask, request, jsonify
from pathlib import Path
from ocr import run_ocr

app = Flask(__name__)
MAX_UPLOAD_MB = 10

app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

def allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

@app.route("/read-meter", methods=["POST"])
def read_meter():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]

    if not file.filename or not allowed(file.filename):
        return jsonify({"error": f"Unsupported file type"}), 400

    result = run_ocr(file.read())
    return jsonify(result), 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, request, render_template
import os
from utils import run_inference

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['audio']

    if file.filename == '':
        return render_template("result.html", status="error", message="❌ No audio file selected.")

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    result = run_inference(path)

    if result['status'] == 'error':
        return render_template("result.html", status="error", message=f"❌ Error: {result['message']}")

    prediction = "REAL" if result['prediction'] > 0.5 else "FAKE"
    confidence = f"{result['prediction']:.4f}"

    return render_template("result.html", status="success", prediction=prediction, confidence=confidence)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

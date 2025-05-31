import numpy as np
import pywt
import librosa
import librosa.feature
from scipy.stats import entropy
from pywt import WaveletPacket
import joblib
import pandas as pd
import logging
import tensorflow as tf
import os

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the scaler and transformer
scaler = joblib.load('robust_scaler1.pkl')
transformer = joblib.load('yeo_johnson_transform.pkl')

# Load and Normalize Audio
def load_audio(filename):
    y, sr = librosa.load(filename, sr=16000)
    logging.info(f"Loaded audio: shape={y.shape}, dtype={y.dtype}, min={y.min()}, max={y.max()}, sr={sr}")
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    return sr, y

# Empirical Wavelet Transform
def ewt_decompose(signal):
    coeffs = pywt.wavedec(signal, 'db4', level=4)
    return coeffs

# Wavelet Packet Transform
def wpt_decompose(signal, wavelet='db4', level=3):
    wp = WaveletPacket(data=signal, wavelet=wavelet, maxlevel=level)
    nodes = [node.path for node in wp.get_level(level)]
    return [wp[node].data for node in nodes]

# Extract Statistical Features
def extract_wavelet_features(ewt_coeffs, wpt_coeffs):
    features = []
    for coeff in ewt_coeffs:
        features += [np.mean(coeff), np.var(coeff), entropy(np.abs(coeff) + 1e-10)]
    for coeff in wpt_coeffs:
        features += [np.mean(coeff), np.var(coeff), entropy(np.abs(coeff) + 1e-10)]
    return features

# Spectral Feature
def extract_spectral_feature(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return 1 - np.std(mfcc) / np.mean(mfcc)

# Pitch Variation
def extract_pitch_variation(y, sr):
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    valid = pitches[magnitudes > np.median(magnitudes)]
    return np.std(valid) / np.mean(valid) if len(valid) > 0 and np.mean(valid) > 0 else 0

# Full Feature Extraction Pipeline
def extract_features(file_path):
    try:
        sr, y = load_audio(file_path)
        if y is None or len(y) == 0 or np.all(y == 0):
            raise ValueError("Audio file is empty or silent.")
        ewt_coeffs = ewt_decompose(y)
        wpt_coeffs = wpt_decompose(y)
        wavelet_feats = extract_wavelet_features(ewt_coeffs, wpt_coeffs)
        spectral_feat = extract_spectral_feature(y, sr)
        pitch_var = extract_pitch_variation(y, sr)
        feature_list = [spectral_feat, pitch_var] + wavelet_feats
        feature_vector = np.array(feature_list).reshape(1, -1)
        return feature_vector.astype(np.float32)
    except Exception as e:
        logging.exception(f"❌ Feature extraction failed for {file_path}")
        return None

# Run Inference
def run_inference(file_path):
    try:
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path="model.tflite")
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Feature extraction
        features = extract_features(file_path)
        if features is None:
            return {"status": "error", "message": "Failed to extract features."}

        # Apply Yeo-Johnson transformation
        features = transformer.transform(features)

        # Robust scaling
        if hasattr(scaler, 'feature_names_in_'):
            col_names = scaler.feature_names_in_
            feature_df = pd.DataFrame(features, columns=col_names)
            features_scaled = scaler.transform(feature_df)
        else:
            features_scaled = scaler.transform(features)
        features_scaled = features_scaled.astype(np.float32)

        # Sanity check for input shape
        if features_scaled.shape[1] != input_details[0]['shape'][1]:
            return {"status": "error", "message": f"Feature shape mismatch: {features_scaled.shape[1]} vs model input {input_details[0]['shape'][1]}"}

        # Inference
        interpreter.set_tensor(input_details[0]['index'], features_scaled)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        prob = float(output[0][0])
        predicted_class = 1 if prob > 0.5 else 0

        return {
            "file": file_path,
            "status": "success",
            "label": "Bona fide" if predicted_class == 1 else "Spoof",
            "prediction": int(predicted_class),
            "confidence": round(prob if predicted_class else 1 - prob, 4)
        }
    except Exception as e:
        logging.exception("❌ Inference failed")
        return {"status": "error", "message": f"Model inference failed: {e}"}

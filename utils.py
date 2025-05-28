import numpy as np
import pywt
import librosa
import librosa.feature
from scipy.stats import entropy
from pywt import WaveletPacket
import joblib
import pandas as pd

# Loading the scaler from a file
scaler =joblib.load('robust_scaler.pkl')

# Loading and Normalize Audio
def load_audio(filename):
    y, sr = librosa.load(filename, sr=16000)  # Use consistent sampling rate
    y = y / np.max(np.abs(y))
    return sr, y

# mpirical Wavelet Transform (EWT)
def ewt_decompose(signal):
    coeffs = pywt.wavedec(signal, 'db4', level=4)
    return coeffs

# Wavelet Packet Transform (WPT)
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

# Spectral Feature (MFCC Consistency)
def extract_spectral_feature(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return 1 - np.std(mfcc) / np.mean(mfcc)

# Pitch Variation
def extract_pitch_variation(y, sr):
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    valid = pitches[magnitudes > np.median(magnitudes)]
    return np.std(valid) / np.mean(valid) if len(valid) > 0 and np.mean(valid) > 0 else 0

# Process 
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

        # Use the same feature names as during scaler fitting
        if hasattr(scaler, 'feature_names_in_'):
            col_names = scaler.feature_names_in_
            feature_df = pd.DataFrame(feature_vector, columns=col_names)
            scaled_vector = scaler.transform(feature_df)
        else:
            scaled_vector = scaler.transform(feature_vector)
        return scaled_vector.astype(np.float32)

    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None


# To Run inference using the model
def run_inference(file_path):
    
    from tensorflow.lite.python.interpreter import Interpreter

    interpreter = Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    features = extract_features(file_path)
    if features is None:
        return {"status": "error", "message": "Failed to extract features from audio."}

    interpreter.set_tensor(input_details[0]['index'], features)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    # Get raw probability
    prob = float(output[0][0])

    # Apply threshold
    predicted_class = 1 if prob > 0.5 else 0

    return {
        "status": "success",
        "prediction": predicted_class,     # 1 = REAL, 0 = FAKE
        "confidence": prob                 # raw sigmoid output
    }


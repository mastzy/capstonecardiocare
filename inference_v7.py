"""
inference_v7.py — CardioCare AI Prediction Module
==================================================
Divisi AI Engineering | CC26-PSU234 | Coding Camp 2026

CARA PAKAI:
    from inference_v7 import predict_cardio

    hasil = predict_cardio({
        "age": 55, "gender": 1, "height": 165, "weight": 82,
        "ap_hi": 145, "ap_lo": 92, "cholesterol": 2,
        "gluc": 1, "smoke": 0, "alco": 0, "active": 1
    })
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import joblib

# =============================================================================
# 1. CUSTOM OBJECTS (Wajib sinkron dengan notebook training)
# =============================================================================

class MedicalFeatureAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1],),
            initializer="ones",
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        return inputs * tf.nn.softmax(self.w)

    def get_config(self):
        return super().get_config()


class ClinicalAsymmetricLoss(tf.keras.losses.Loss):
    def __init__(self, fn_weight=1.5, **kwargs):
        super().__init__(**kwargs)
        self.fn_weight = fn_weight

    def call(self, y_true, y_pred):
        y_true   = tf.cast(y_true, tf.float32)
        epsilon  = tf.keras.backend.epsilon()
        y_pred   = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        loss_pos = -y_true * tf.math.log(y_pred) * self.fn_weight
        loss_neg = -(1.0 - y_true) * tf.math.log(1.0 - y_pred)
        return tf.reduce_mean(loss_pos + loss_neg)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'fn_weight': self.fn_weight})
        return cfg


class BinaryMAE(tf.keras.metrics.Metric):
    """
    Custom metric yang dipakai saat training — wajib ada di CUSTOM_OBJECTS
    agar model bisa di-load tanpa error di semua versi TensorFlow.
    """
    def __init__(self, threshold=0.40, name='binary_mae', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.total_mae = self.add_weight(name='total_mae', initializer='zeros')
        self.count     = self.add_weight(name='count',     initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true        = tf.cast(y_true, tf.float32)
        y_pred_binary = tf.cast(y_pred >= self.threshold, tf.float32)
        self.total_mae.assign_add(tf.reduce_mean(tf.abs(y_true - y_pred_binary)))
        self.count.assign_add(1.0)

    def result(self):
        return self.total_mae / (self.count + tf.keras.backend.epsilon())

    def reset_state(self):
        self.total_mae.assign(0.0)
        self.count.assign(0.0)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'threshold': self.threshold})
        return cfg


# FIX: BinaryMAE wajib ada — tanpa ini load_model() bisa gagal di beberapa versi TF
CUSTOM_OBJECTS = {
    'MedicalFeatureAttention': MedicalFeatureAttention,
    'ClinicalAsymmetricLoss' : ClinicalAsymmetricLoss,
    'BinaryMAE'              : BinaryMAE,
}

# =============================================================================
# 2. PATH KONFIGURASI
# =============================================================================
_BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(_BASE_DIR, 'cardiocare_model_v7.keras')
SCALER_PATH = os.path.join(_BASE_DIR, 'scaler_v7.pkl')
THRESHOLD   = 0.40  # Threshold medis optimal v7 — jangan diubah tanpa koordinasi AI Eng

# =============================================================================
# 3. LAZY LOADING — model dimuat SEKALI saat pertama kali predict_cardio() dipanggil
#    (bukan saat file di-import, agar app tidak crash jika file belum siap)
# =============================================================================
_model  = None
_scaler = None

def _load_artifacts():
    global _model, _scaler
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model tidak ditemukan: {MODEL_PATH}\n"
                "Pastikan cardiocare_model_v7.keras ada di folder yang sama."
            )
        _model = tf.keras.models.load_model(MODEL_PATH, custom_objects=CUSTOM_OBJECTS)

    if _scaler is None:
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(
                f"Scaler tidak ditemukan: {SCALER_PATH}\n"
                "Pastikan scaler_v7.pkl ada di folder yang sama."
            )
        _scaler = joblib.load(SCALER_PATH)

# =============================================================================
# 4. VALIDASI INPUT
# =============================================================================
def _validate(data: dict) -> None:
    required = ["age", "gender", "height", "weight", "ap_hi", "ap_lo",
                "cholesterol", "gluc", "smoke", "alco", "active"]
    missing = [f for f in required if f not in data]
    if missing:
        raise ValueError(f"Field wajib tidak ada: {missing}")

    if not (18 <= float(data["age"]) <= 100):
        raise ValueError("age harus antara 18–100 tahun")
    if int(data["gender"]) not in [1, 2]:
        raise ValueError("gender: 1=Perempuan, 2=Laki-laki")
    if not (140 <= float(data["height"]) <= 210):
        raise ValueError("height harus antara 140–210 cm")
    if not (40 <= float(data["weight"]) <= 200):
        raise ValueError("weight harus antara 40–200 kg")
    if not (60 <= float(data["ap_hi"]) <= 250):
        raise ValueError("ap_hi harus antara 60–250 mmHg")
    if not (40 <= float(data["ap_lo"]) <= 200):
        raise ValueError("ap_lo harus antara 40–200 mmHg")
    if float(data["ap_hi"]) <= float(data["ap_lo"]):
        raise ValueError("ap_hi (sistolik) harus lebih besar dari ap_lo (diastolik)")
    if int(data["cholesterol"]) not in [1, 2, 3]:
        raise ValueError("cholesterol: 1=Normal, 2=Tinggi, 3=Sangat Tinggi")
    if int(data["gluc"]) not in [1, 2, 3]:
        raise ValueError("gluc: 1=Normal, 2=Tinggi, 3=Sangat Tinggi")
    if int(data["smoke"]) not in [0, 1]:
        raise ValueError("smoke: 0=Tidak, 1=Ya")
    if int(data["alco"]) not in [0, 1]:
        raise ValueError("alco: 0=Tidak, 1=Ya")
    if int(data["active"]) not in [0, 1]:
        raise ValueError("active: 0=Tidak Aktif, 1=Aktif")

# =============================================================================
# 5. FEATURE ENGINEERING — 11 field raw → 24 fitur model
# =============================================================================
def _engineer_features(data: dict) -> np.ndarray:
    age  = float(data["age"])      # TAHUN — bukan hari
    h    = float(data["height"])
    w    = float(data["weight"])
    ap   = float(data["ap_hi"])
    apl  = float(data["ap_lo"])
    chol = int(data["cholesterol"])
    gluc = int(data["gluc"])

    # Fitur turunan dasar
    bmi             = w / (h / 100) ** 2
    pulse_pressure  = ap - apl
    unhealthy_score = float(data["smoke"]) + float(data["alco"]) + (1.0 - float(data["active"]))

    # OHE: cholesterol
    chol_1, chol_2, chol_3 = float(chol == 1), float(chol == 2), float(chol == 3)

    # OHE: gluc
    gluc_1, gluc_2, gluc_3 = float(gluc == 1), float(gluc == 2), float(gluc == 3)

    # OHE: bp_category
    # FIX: logika ini HARUS identik dengan notebook training
    #   Normal       : ap < 120 AND apl < 80
    #   Pre-hipertensi: 120 <= ap < 140 OR 80 <= apl < 90   ← OR bukan AND
    #   Hipertensi   : selain di atas
    if ap < 120 and apl < 80:
        bp_cat = 1
    elif 120 <= ap < 140 or 80 <= apl < 90:   # ← OR, bukan (120<=ap<130 AND apl<80)
        bp_cat = 2
    else:
        bp_cat = 3

    bp_1, bp_2, bp_3 = float(bp_cat == 1), float(bp_cat == 2), float(bp_cat == 3)

    # Fitur interaksi klinis
    ap_hi_x_age = ap  * age
    pp_x_age    = pulse_pressure * age
    bmi_x_ap_hi = bmi * ap

    return np.array([[
        age, float(data["gender"]), h, w, ap, apl,
        float(data["smoke"]), float(data["alco"]), float(data["active"]),
        bmi, pulse_pressure, unhealthy_score,
        chol_1, chol_2, chol_3,
        gluc_1, gluc_2, gluc_3,
        bp_1, bp_2, bp_3,
        ap_hi_x_age, pp_x_age, bmi_x_ap_hi
    ]], dtype=np.float32)

# =============================================================================
# 6. FUNGSI UTAMA
# =============================================================================
def predict_cardio(data: dict, threshold: float = THRESHOLD) -> dict:
    """
    Prediksi risiko penyakit kardiovaskular.

    Parameters
    ----------
    data : dict  — 11 field pasien (lihat contoh di bawah)
    threshold : float — default 0.40 (threshold medis optimal)

    Returns
    -------
    dict:
        label            : 0 atau 1
        probability      : float 0.0–1.0
        category         : 'BERISIKO TINGGI' atau 'RISIKO RENDAH/AMAN'
        recommendation   : teks rekomendasi
        threshold_used   : threshold yang dipakai
        top_risk_factors : list faktor risiko teridentifikasi

    Example
    -------
    >>> hasil = predict_cardio({
    ...     "age": 55, "gender": 1, "height": 165, "weight": 82,
    ...     "ap_hi": 145, "ap_lo": 92, "cholesterol": 2,
    ...     "gluc": 1, "smoke": 0, "alco": 0, "active": 1
    ... })
    """
    _load_artifacts()
    _validate(data)

    X        = _engineer_features(data)
    X_scaled = _scaler.transform(X)
    prob     = float(_model.predict(X_scaled, verbose=0)[0][0])
    label    = 1 if prob >= threshold else 0

    # Identifikasi faktor risiko
    bmi_val = float(data["weight"]) / (float(data["height"]) / 100) ** 2
    risk_factors = []
    if float(data["ap_hi"]) >= 140:
        risk_factors.append("Hipertensi (tekanan sistolik tinggi)")
    if bmi_val >= 30:
        risk_factors.append(f"Obesitas (BMI = {bmi_val:.1f})")
    elif bmi_val >= 25:
        risk_factors.append(f"Kelebihan berat badan (BMI = {bmi_val:.1f})")
    if int(data["cholesterol"]) >= 2:
        risk_factors.append("Kolesterol di atas normal")
    if int(data["gluc"]) >= 2:
        risk_factors.append("Glukosa di atas normal")
    if float(data["age"]) >= 55:
        risk_factors.append("Usia kategori risiko tinggi (>= 55 tahun)")
    if int(data["smoke"]):
        risk_factors.append("Perokok aktif")
    if int(data["alco"]):
        risk_factors.append("Konsumsi alkohol")
    if not int(data["active"]):
        risk_factors.append("Kurang aktivitas fisik")

    return {
        "label"           : label,
        "probability"     : round(prob, 4),
        "category"        : "BERISIKO TINGGI" if label == 1 else "RISIKO RENDAH/AMAN",
        "recommendation"  : (
            "Segera konsultasikan kondisi Anda ke dokter atau tenaga medis."
            if label == 1 else
            "Pertahankan gaya hidup sehat dan lakukan pemeriksaan rutin."
        ),
        "threshold_used"  : threshold,
        "top_risk_factors": risk_factors,
    }

# =============================================================================
# 7. QUICK TEST — jalankan: python inference_v7.py
# =============================================================================
if __name__ == "__main__":
    print("=" * 55)
    print("  CardioCare v7 — Quick Test")
    print("=" * 55)

    tests = [
        ("Pasien Berisiko", {
            "age": 58, "gender": 2, "height": 170, "weight": 95,
            "ap_hi": 150, "ap_lo": 95, "cholesterol": 3,
            "gluc": 2, "smoke": 1, "alco": 0, "active": 0
        }),
        ("Pasien Sehat", {
            "age": 30, "gender": 1, "height": 162, "weight": 55,
            "ap_hi": 115, "ap_lo": 75, "cholesterol": 1,
            "gluc": 1, "smoke": 0, "alco": 0, "active": 1
        }),
    ]

    for label, sample in tests:
        print(f"\n[Test: {label}]")
        try:
            h = predict_cardio(sample)
            print(f"  Probabilitas    : {h['probability'] * 100:.2f}%")
            print(f"  Hasil           : {h['category']}")
            print(f"  Rekomendasi     : {h['recommendation']}")
            if h['top_risk_factors']:
                print(f"  Faktor Risiko   :")
                for f in h['top_risk_factors']:
                    print(f"    - {f}")
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n" + "=" * 55)
    print("  Test selesai. Siap diintegrasikan ke API.")
    print("=" * 55)

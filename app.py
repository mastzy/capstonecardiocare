"""
app.py — CardioCare AI Service (FastAPI)
========================================
Divisi FS | CC26-PSU234 | Coding Camp 2026

Install:
    pip install fastapi uvicorn python-dotenv google-genai
    pip install tensorflow scikit-learn pandas numpy joblib

Jalankan:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    POST /predict        ← prediksi utama
    GET  /health         ← cek status server & model
    GET  /threshold      ← info threshold aktif
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from google import genai

# FIX: import predict_cardio dari inference_v7 — jangan duplikasi feature engineering
# Semua logika preprocessing, validasi, dan prediksi sudah ada di sini
from inference_v7 import predict_cardio

# =============================================================================
# 1. INISIALISASI APP
# =============================================================================
app = FastAPI(
    title="CardioCare AI Service",
    description="REST API untuk prediksi risiko jantung v7 + rekomendasi Gemini AI",
    version="7.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Sesuaikan dengan domain frontend di production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inisialisasi Gemini Client (baca GEMINI_API_KEY dari environment / .env)
try:
    ai_client = genai.Client()
except Exception:
    ai_client = None  # Fallback: rekomendasi statis jika key belum di-set

# =============================================================================
# 2. SCHEMA VALIDASI INPUT (Pydantic)
# =============================================================================
class PatientInput(BaseModel):
    age         : float = Field(..., ge=18, le=100,  description="Usia dalam TAHUN (contoh: 55) — bukan hari")
    gender      : int   = Field(..., ge=1,  le=2,    description="1=Perempuan, 2=Laki-laki")
    height      : float = Field(..., ge=140, le=210, description="Tinggi badan dalam cm")
    weight      : float = Field(..., ge=40,  le=200, description="Berat badan dalam kg")
    ap_hi       : float = Field(..., ge=60,  le=250, description="Tekanan darah sistolik (mmHg)")
    ap_lo       : float = Field(..., ge=40,  le=200, description="Tekanan darah diastolik (mmHg)")
    cholesterol : int   = Field(..., ge=1,  le=3,    description="1=Normal, 2=Tinggi, 3=Sangat Tinggi")
    gluc        : int   = Field(..., ge=1,  le=3,    description="1=Normal, 2=Tinggi, 3=Sangat Tinggi")
    smoke       : int   = Field(..., ge=0,  le=1,    description="0=Tidak Merokok, 1=Merokok")
    alco        : int   = Field(..., ge=0,  le=1,    description="0=Tidak, 1=Konsumsi Alkohol")
    active      : int   = Field(..., ge=0,  le=1,    description="0=Tidak Aktif, 1=Aktif Fisik")

    # Validasi relasi ap_hi > ap_lo yang tidak bisa ditangkap Field()
    @validator('ap_lo')
    def sistolik_harus_lebih_besar(cls, ap_lo, values):
        if 'ap_hi' in values and values['ap_hi'] <= ap_lo:
            raise ValueError("ap_hi (sistolik) harus lebih besar dari ap_lo (diastolik)")
        return ap_lo

    class Config:
        schema_extra = {
            "example": {
                "age": 55, "gender": 1, "height": 165, "weight": 82,
                "ap_hi": 145, "ap_lo": 92, "cholesterol": 2,
                "gluc": 1, "smoke": 0, "alco": 0, "active": 1
            }
        }

# =============================================================================
# 3. GEMINI AI — REKOMENDASI MEDIS GENERATIF
# =============================================================================
def generate_medical_recommendation(
    age: float, ap_hi: float, ap_lo: float,
    category: str, probability: float,
    risk_factors: list
) -> str:
    """
    Generate rekomendasi teks bergaya dokter menggunakan Gemini AI.
    Jika API key tidak tersedia, kembalikan rekomendasi statis.
    """
    if not ai_client:
        if category == "BERISIKO TINGGI":
            return ("1. Segera konsultasikan ke dokter spesialis jantung.\n"
                    "2. Monitor tekanan darah secara rutin setiap hari.\n"
                    "3. Kurangi konsumsi garam dan makanan berlemak.")
        return ("1. Pertahankan pola makan sehat dan seimbang.\n"
                "2. Lakukan olahraga kardio minimal 30 menit per hari.\n"
                "3. Lakukan pemeriksaan kesehatan rutin setiap 6 bulan.")

    faktor = "\n".join([f"- {f}" for f in risk_factors]) if risk_factors else "- Tidak ada faktor risiko signifikan teridentifikasi"

    prompt = f"""Anda adalah asisten dokter spesialis jantung profesional.
Berikan rekomendasi tindakan medis yang singkat, padat, dan taktis (3 poin) untuk pasien berikut:

- Usia          : {age:.0f} tahun
- Tekanan Darah : {int(ap_hi)}/{int(ap_lo)} mmHg
- Hasil AI      : {category} (Probabilitas: {probability * 100:.1f}%)
- Faktor Risiko :
{faktor}

Gunakan bahasa Indonesia yang santun, edukatif, dan mudah dipahami pasien awam.
Format: nomor poin, kalimat singkat dan langsung ke tindakan."""

    try:
        response = ai_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text.strip()
    except Exception:
        return "Gagal terhubung ke Gemini. Silakan konsultasikan hasil ke dokter terdekat."

# =============================================================================
# 4. ENDPOINTS
# =============================================================================

@app.post("/predict", summary="Prediksi Risiko Kardiovaskular")
async def predict(patient: PatientInput):
    """
    Terima 11 field data pasien, kembalikan prediksi risiko + rekomendasi Gemini.

    **Contoh request body:**
    ```json
    {
        "age": 55, "gender": 1, "height": 165, "weight": 82,
        "ap_hi": 145, "ap_lo": 92, "cholesterol": 2,
        "gluc": 1, "smoke": 0, "alco": 0, "active": 1
    }
    ```
    """
    # FIX: gunakan predict_cardio() dari inference_v7 — tidak ada duplikasi feature engineering
    raw_data = patient.dict()

    try:
        hasil = predict_cardio(raw_data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Konfigurasi server error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Layanan sementara tidak tersedia.")

    # Generate rekomendasi Gemini AI
    ai_advice = generate_medical_recommendation(
        age         = patient.age,
        ap_hi       = patient.ap_hi,
        ap_lo       = patient.ap_lo,
        category    = hasil["category"],
        probability = hasil["probability"],
        risk_factors= hasil["top_risk_factors"],
    )

    return {
        "status": "success",
        "prediction": {
            "label"          : hasil["label"],
            "probability"    : hasil["probability"],
            "category"       : hasil["category"],
            "threshold_used" : hasil["threshold_used"],
        },
        "clinical_analysis": {
            "top_risk_factors": hasil["top_risk_factors"],
            "recommendation"  : hasil["recommendation"],
        },
        "ai_expert_recommendation": ai_advice,
    }


@app.get("/health", summary="Cek Status Server & Model")
async def health():
    """Verifikasi bahwa server berjalan dan model sudah ter-load."""
    try:
        # Prediksi dummy untuk verifikasi model loaded
        predict_cardio({
            "age": 40, "gender": 1, "height": 160, "weight": 60,
            "ap_hi": 120, "ap_lo": 80, "cholesterol": 1,
            "gluc": 1, "smoke": 0, "alco": 0, "active": 1
        })
        return {
            "status"   : "ok",
            "model"    : "cardiocare_v7",
            "threshold": 0.40,
            "message"  : "Server dan model siap digunakan."
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model belum siap: {str(e)}")


@app.get("/threshold", summary="Info Threshold Aktif")
async def threshold_info():
    """Informasi threshold yang digunakan dan alasan klinisnya."""
    return {
        "threshold_aktif": 0.40,
        "keterangan"     : (
            "Threshold medis optimal — memaksimalkan Recall (85.94%) "
            "untuk meminimalkan pasien berisiko yang terlewat (False Negative)."
        ),
        "perbandingan"   : {
            "threshold_0.40": {"recall": "85.94%", "accuracy": "68.94%", "fn": 953},
            "threshold_0.50": {"recall": "75.90%", "accuracy": "72.26%", "fn": 1633},
        },
        "catatan": "Perubahan threshold harus koordinasi dengan Divisi AI Engineering."
    }


# =============================================================================
# 5. ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    print("CardioCare AI Service v7")
    print("  POST /predict   ← prediksi utama")
    print("  GET  /health    ← cek status")
    print("  GET  /threshold ← info threshold")
    print("  Docs: http://localhost:8000/docs")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

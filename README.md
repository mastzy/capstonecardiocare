🫀 CardioCare AI
Sistem prediksi risiko penyakit kardiovaskular berbasis deep learning (TensorFlow) yang dikembangkan sebagai Capstone Project Coding Camp 2026.

📋 Deskripsi Proyek
CardioCare AI dirancang sebagai alat bantu skrining awal untuk mendeteksi potensi risiko penyakit kardiovaskular. Dengan mengutamakan Sensitivitas (Recall) sebesar 85.94% menggunakan threshold medis 0.40, sistem ini meminimalkan risiko False Negative (pasien berisiko yang tidak terdeteksi).

📂 Struktur Repositori
Struktur ini diatur agar alur data antara model, preprocessing, dan aplikasi API tetap terisolasi dan mudah dipelihara:

Plaintext
cardio-care-project/
├── .gitignore            # Pengabaian file sampah & data sensitif (.env)
├── README.md             # Dokumentasi proyek
├── requirements.txt      # Daftar dependencies library
├── app.py                # Main API (FastAPI) — pintu masuk aplikasi
└── model/                # Folder khusus model
    ├── cardiocare_model_v7.keras  # File model TensorFlow utama
    ├── scaler_v7.pkl              # Scaler untuk normalisasi input
    ├── model_metadata.pkl         # Konfigurasi & versi model
    └── inference_v7.py            # Fungsi predict_cardio (API integrasi)
⚙️ Cara Penggunaan
Persiapan: Pastikan Python 3.9+ telah terpasang.

Instalasi: Masuk ke folder proyek dan jalankan perintah:

Bash
pip install -r requirements.txt
Menjalankan API: Jalankan server FastAPI dengan perintah:

Bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
Endpoint:

POST /predict: Kirim data pasien (11 field) untuk prediksi.

GET /health: Cek status kesiapan model.

GET /threshold: Lihat informasi threshold medis yang digunakan.

🧠 Analisis Teknis
Proyek ini mengimplementasikan Custom Layer MedicalFeatureAttention untuk memberikan bobot lebih pada fitur klinis yang krusial.

Proses integrasi ini dirancang agar tim Frontend tidak perlu memahami kerumitan deep learning di balik layar; cukup kirim data mentah, dan model akan mengembalikan hasil prediksi yang sudah tervalidasi.

⚖️ Disclaimer Medis
Model ini adalah alat bantu skrining awal dan bukan pengganti diagnosis medis profesional. Setiap hasil prediksi 'BERISIKO TINGGI' wajib dikonfirmasi oleh tenaga medis melalui pemeriksaan klinis lebih lanjut.

Catatan Tambahan:
Keamanan: Pastikan file .env (berisi API Key Gemini jika digunakan) tidak pernah di-push ke repositori publik.

Integrasi: inference_v7.py sudah menangani preprocessing otomatis. Divisi Frontend hanya perlu mengirimkan dictionary berisi 11 data klinis yang tepat sesuai dokumentasi integrasi.

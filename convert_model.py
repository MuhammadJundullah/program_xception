import os
import shutil
import zipfile
from pathlib import Path

print("--- Memulai ekstraksi bobot mentah dari file .keras ---")

BASE_DIR = Path(__file__).resolve().parent
keras_model_path = BASE_DIR / "xception_deepfake_best.keras"
extract_dir = BASE_DIR / "temp_extracted_keras"
target_weights_path = BASE_DIR / "xception_deepfake_weights.weights.h5"

if not keras_model_path.exists():
    print(f"ERROR: File tidak ditemukan: {keras_model_path}")
    raise SystemExit(1)

try:
    # 1. Ekstrak file .keras sebagai file ZIP biasa.
    with zipfile.ZipFile(keras_model_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    print("OK: Berhasil membongkar arsip .keras")

    # 2. Cari file bobot di dalam struktur Keras 3.
    source_weights_path = extract_dir / "model.weights.h5"

    if source_weights_path.exists():
        # 3. Salin dan rename ke folder project utama.
        shutil.copy(source_weights_path, target_weights_path)
        print(f"OK: Berhasil ekstrak bobot. File tersimpan di: {target_weights_path}")
    else:
        print("ERROR: File 'model.weights.h5' tidak ditemukan di dalam arsip.")
        print("Isi file arsip:", os.listdir(extract_dir))

except Exception as exc:
    print(f"ERROR: Gagal mengekstrak bobot: {exc}")
    raise SystemExit(1) from exc

finally:
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
        print("OK: Membersihkan file sementara selesai.")

import zipfile
import os
import shutil

print("--- Memulai Ekstraksi Bobot Mentah dari File .keras ---")

keras_model_path = "./xception_deepfake_best.keras"
extract_dir = "./temp_extracted_keras"
target_weights_name = "./xception_deepfake_weights.weights.h5"

if not os.path.exists(keras_model_path):
    print(f"❌ File tidak ditemukan: {keras_model_path}")
    exit()

try:
    # 1. Ekstrak file .keras sebagai file ZIP biasa
    with zipfile.ZipFile(keras_model_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("✓ Berhasil membongkar arsip .keras")

    # 2. Cari file bobot di dalam struktur Keras 3
    # Di Keras 3, file bobot biasanya berada di: model.weights.h5
    source_weights_path = os.path.join(extract_dir, "model.weights.h5")

    if os.path.exists(source_weights_path):
        # 3. Pindahkan dan rename ke folder project utama
        shutil.copy(source_weights_path, target_weights_name)
        print(f"✓ BERHASIL EKSTRAK BOBOT! File tersimpan di: {target_weights_name}")
    else:
        print("❌ File 'model.weights.h5' tidak ditemukan di dalam arsip.")
        # Cetak isi folder untuk inspeksi jika strukturnya berbeda
        print("Isi file arsip:", os.listdir(extract_dir))

except Exception as e:
    print(f"❌ Gagal mengekstrak bobot: {e}")

finally:
    # Bersihkan folder sampah hasil ekstraksi
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
        print("✓ Membersihkan file sementara selesai.")
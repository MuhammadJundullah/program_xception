import sys
from pathlib import Path

import cv2
import numpy as np
import ultralytics
from mss import mss
from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
FACE_MODEL_PATH = BASE_DIR / "yolov8n-face.pt"
CLASSIFIER_CANDIDATES = (
    BASE_DIR / "xception_deepfake_best.keras",
    BASE_DIR / "xception_deepfake_weights.weights.h5",
)


def resolve_classifier_path():
    for path in CLASSIFIER_CANDIDATES:
        if path.exists():
            return path
    searched = ", ".join(str(path.name) for path in CLASSIFIER_CANDIDATES)
    raise FileNotFoundError(
        f"Model classifier tidak ditemukan. Cari salah satu file ini di folder project: {searched}"
    )


def build_xception_architecture(tf):
    base_model = tf.keras.applications.Xception(
        weights=None,
        include_top=False,
        input_shape=(299, 299, 3),
    )
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs=base_model.input, outputs=outputs)

# ok 
def load_classifier_model(classifier_path, tf):
    classifier_name = classifier_path.name

    if classifier_name.endswith(".weights.h5"):
        try:
            model = build_xception_architecture(tf)
            model.load_weights(str(classifier_path))
            return model
        except Exception as exc:
            raise RuntimeError(
                f"Gagal memuat bobot mentah ke arsitektur Xception: {exc}"
            ) from exc

    loaders = []
    try:
        import keras

        loaders.append((f"Keras Standalone v{keras.__version__}", keras.models.load_model))
    except Exception:
        pass

    loaders.append((f"tf.keras v{tf.__version__}", tf.keras.models.load_model))

    load_errors = []
    for loader_name, load_model_fn in loaders:
        try:
            return load_model_fn(str(classifier_path), compile=False)
        except Exception as exc:
            message = str(exc)
            load_errors.append(f"{loader_name}: {message}")
            if "Could not deserialize class" in message or "keras.src.models" in message:
                raise RuntimeError(
                    "Model `.keras` tidak kompatibel dengan runtime Keras yang aktif.\n"
                    "Biasanya ini terjadi karena model dibuat di Keras 3 tetapi dibuka di TensorFlow/Keras 2.x.\n"
                    "Perbaikan: upgrade `tensorflow` dan `keras`, atau export ulang model ke format yang kompatibel."
                ) from exc

    combined_errors = "\n".join(load_errors)
    raise RuntimeError(
        f"Gagal memuat model classifier '{classifier_name}'.\nDetail percobaan:\n{combined_errors}"
    )


def validate_face_model_environment():
    if not FACE_MODEL_PATH.exists():
        raise FileNotFoundError(f"File model wajah tidak ditemukan: {FACE_MODEL_PATH.name}")

    if FACE_MODEL_PATH.name.startswith("yolov11"):
        try:
            from ultralytics.nn.modules import block as yolo_block
        except Exception as exc:
            raise RuntimeError(f"Gagal memeriksa modul Ultralytics: {exc}") from exc

        if not hasattr(yolo_block, "C3k2"):
            raise RuntimeError(
                "Model 'yolov11n-face.pt' membutuhkan versi Ultralytics yang lebih baru. "
                f"Versi terpasang saat ini: {ultralytics.__version__}. "
                "Upgrade package `ultralytics` atau gunakan weight yang cocok."
            )


def load_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        message = str(exc)
        if "runtime_version" in message and "google.protobuf" in message:
            raise RuntimeError(
                "TensorFlow gagal diimport karena protobuf tidak kompatibel.\n"
                "Perbaiki environment dengan menjalankan:\n"
                "pip install --upgrade tensorflow keras protobuf"
            ) from exc
        raise RuntimeError(f"Gagal import TensorFlow: {exc}") from exc
    return tf


def annotate_frame(
    frame,
    face_detector,
    deepfake_classifier,
    tf,
    input_size=(299, 299),
    confidence_threshold=0.25,
):
    annotated = frame.copy()
    results = face_detector(annotated, verbose=False)[0]

    for box in results.boxes:
        confidence = float(box.conf[0]) if box.conf is not None else 0.0
        if confidence < confidence_threshold:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        h, w, _ = annotated.shape
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

        face_roi = annotated[y1:y2, x1:x2]
        if face_roi.size == 0:
            continue

        face_resized = cv2.resize(face_roi, input_size)
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_array = tf.keras.preprocessing.image.img_to_array(face_rgb)
        face_array = np.expand_dims(face_array, axis=0)
        face_array = face_array / 255.0

        prediction_raw = deepfake_classifier.predict(face_array, verbose=0)
        prediction = float(np.asarray(prediction_raw).squeeze())
        prediction = max(0.0, min(1.0, prediction))

        if prediction > 0.5:
            label = f"DEEPFAKE: {prediction * 100:.1f}%"
            color = (0, 0, 255)
        else:
            label = f"REAL: {(1 - prediction) * 100:.1f}%"
            color = (0, 255, 0)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    return annotated


class ScreenProcessorThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    error_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.face_detector = None
        self.deepfake_classifier = None
        self.tf = None
        self.input_size = (299, 299)

    def _load_models(self):
        validate_face_model_environment()
        classifier_path = resolve_classifier_path()
        self.tf = load_tensorflow()
        self.face_detector = YOLO(str(FACE_MODEL_PATH))
        self.deepfake_classifier = load_classifier_model(classifier_path, self.tf)

    def run(self):
        try:
            self._load_models()

            with mss() as sct:
                if len(sct.monitors) < 2:
                    raise RuntimeError("Monitor untuk screen capture tidak terdeteksi.")

                monitor = sct.monitors[1]

                while self._run_flag:
                    sct_img = sct.grab(monitor)
                    frame = np.array(sct_img)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    annotated = annotate_frame(
                        frame,
                        self.face_detector,
                        self.deepfake_classifier,
                        self.tf,
                        self.input_size,
                    )
                    self.change_pixmap_signal.emit(annotated)
        except Exception as exc:
            self._run_flag = False
            self.error_signal.emit(str(exc))

    def stop(self):
        self._run_flag = False
        self.wait()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Screen Deepfake Detector")
        self.setGeometry(100, 100, 1024, 768)

        self.thread = None
        self.current_mode = None
        self.tf = None
        self.face_detector = None
        self.deepfake_classifier = None
        self.input_size = (299, 299)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.stack = QStackedWidget(self)
        self.layout.addWidget(self.stack)

        self.menu_page = QWidget()
        self.menu_layout = QVBoxLayout(self.menu_page)
        self.menu_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.title_label = QLabel("Pilih Mode Deteksi", self)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.menu_layout.addWidget(self.title_label)

        self.upload_mode_btn = QPushButton("Upload Gambar", self)
        self.upload_mode_btn.clicked.connect(self.open_upload_mode)
        self.menu_layout.addWidget(self.upload_mode_btn)

        self.realtime_mode_btn = QPushButton("Realtime", self)
        self.realtime_mode_btn.clicked.connect(self.open_realtime_mode)
        self.menu_layout.addWidget(self.realtime_mode_btn)

        self.viewer_page = QWidget()
        self.viewer_layout = QVBoxLayout(self.viewer_page)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.viewer_layout.addWidget(self.image_label)

        self.button_row = QHBoxLayout()

        self.back_btn = QPushButton("Kembali ke Menu", self)
        self.back_btn.clicked.connect(self.back_to_menu)
        self.button_row.addWidget(self.back_btn)

        self.upload_btn = QPushButton("Pilih Gambar", self)
        self.upload_btn.clicked.connect(self.select_image)
        self.button_row.addWidget(self.upload_btn)

        self.toggle_btn = QPushButton("Start Screening", self)
        self.toggle_btn.clicked.connect(self.toggle_screening)
        self.button_row.addWidget(self.toggle_btn)

        self.viewer_layout.addLayout(self.button_row)

        self.stack.addWidget(self.menu_page)
        self.stack.addWidget(self.viewer_page)
        self.stack.setCurrentWidget(self.menu_page)

    def ensure_models_loaded(self):
        if self.face_detector is not None and self.deepfake_classifier is not None and self.tf is not None:
            return

        validate_face_model_environment()
        classifier_path = resolve_classifier_path()
        self.tf = load_tensorflow()
        self.face_detector = YOLO(str(FACE_MODEL_PATH))
        self.deepfake_classifier = load_classifier_model(classifier_path, self.tf)

    def open_upload_mode(self):
        self.stop_screening_if_needed()
        self.current_mode = "upload"
        self.image_label.clear()
        self.upload_btn.show()
        self.toggle_btn.hide()
        self.stack.setCurrentWidget(self.viewer_page)

    def open_realtime_mode(self):
        self.stop_screening_if_needed()
        self.current_mode = "realtime"
        self.image_label.clear()
        self.upload_btn.hide()
        self.toggle_btn.show()
        self.toggle_btn.setText("Start Screening")
        self.stack.setCurrentWidget(self.viewer_page)

    def back_to_menu(self):
        self.stop_screening_if_needed()
        self.current_mode = None
        self.image_label.clear()
        self.stack.setCurrentWidget(self.menu_page)

    def stop_screening_if_needed(self):
        if self.thread is not None and self.thread.isRunning():
            self.thread.stop()
        self.thread = None
        self.toggle_btn.setText("Start Screening")

    def toggle_screening(self):
        if self.current_mode != "realtime":
            return

        if self.thread is not None and self.thread.isRunning():
            self.stop_screening_if_needed()
            self.image_label.clear()
            return

        self.toggle_btn.setText("Stop Screening")
        self.thread = ScreenProcessorThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.error_signal.connect(self.handle_thread_error)
        self.thread.start()

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Pilih Gambar",
            str(BASE_DIR),
            "Image Files (*.png *.jpg *.jpeg *.bmp *.webp)",
        )
        if not file_path:
            return

        try:
            self.ensure_models_loaded()
            file_bytes = np.fromfile(file_path, dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError("File gambar tidak bisa dibaca.")

            annotated = annotate_frame(
                frame,
                self.face_detector,
                self.deepfake_classifier,
                self.tf,
                self.input_size,
            )
            self.update_image(annotated)
        except Exception as exc:
            QMessageBox.critical(self, "Model Error", str(exc))

    def update_image(self, cv_img):
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        qt_image = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
        scaled = qt_image.scaled(
            self.image_label.width(),
            self.image_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
        )
        self.image_label.setPixmap(QPixmap.fromImage(scaled))

    def handle_thread_error(self, message):
        self.stop_screening_if_needed()
        self.image_label.clear()
        QMessageBox.critical(self, "Model Error", message)

    def closeEvent(self, event):
        self.stop_screening_if_needed()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

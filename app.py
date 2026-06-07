"""
app.py — MainWindow dan entry point aplikasi Xception Deepfake Detector + SVM Classifier.

Struktur file:
  models.py   — logic model (loading, inference, annotation)
  threads.py  — QThread worker untuk real-time screen capture
  widgets.py  — custom widget (IntroLogoWidget, UploadDropArea, DetectionResultPanel)
  styles.py   — stylesheet Qt seluruh aplikasi
  app.py      — MainWindow + entry point (file ini)
"""

import ctypes
import sys
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from models import (
    BASE_DIR,
    annotate_frame,
    build_result_summary,
    configure_tensorflow_device,
    load_classifier_model,
    load_tensorflow,
    require_torch_gpu,
    resolve_classifier_path,
    validate_face_model_environment,
    FACE_MODEL_PATH,
)
from styles import APP_STYLESHEET
from threads import CameraProcessorThread, ScreenProcessorThread, VideoProcessorThread
from widgets import DetectionResultPanel, IntroLogoWidget, IntroVideoLabel, UploadDropArea

# ── Konstanta Tambahan Untuk SVM ──────────────────────────────────────────────
INTRO_VIDEO_PATH = BASE_DIR / "animasi_deepfake.mp4"
INTRO_LOGO_PATH  = BASE_DIR / "logo_intro.png.png"

WDA_NONE = 0x00000000
WDA_EXCLUDEFROMCAPTURE = 0x00000011


# ── Windows capture-exclusion helper ─────────────────────────────────────────

def set_window_excluded_from_capture(window, excluded: bool) -> bool:
    if not sys.platform.startswith("win"):
        return True
    hwnd = int(window.winId())
    affinity = WDA_EXCLUDEFROMCAPTURE if excluded else WDA_NONE
    return bool(ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, affinity))


# ── MainWindow ────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Xception Deepfake Detector")
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, False)
        self.setGeometry(150, 150, 1000, 680)
        self.setMinimumSize(880, 600)

        # State Modifikasi
        self.thread = None
        self.video_thread = None
        self.camera_thread = None
        self.current_mode = None
        self.tf = None
        self.face_detector = None
        self.deepfake_classifier = None  # Full Xception .keras model
        
        self.input_size = (299, 299)
        self.yolo_device = None
        self.tf_device = "CPU"
        self.capture_exclusion_enabled = False
        self._pending_image_path = None

        # Root
        self.central_widget = QWidget()
        self.central_widget.setObjectName("AppRoot")
        self.setCentralWidget(self.central_widget)
        self._root_layout = QVBoxLayout(self.central_widget)
        self._root_layout.setContentsMargins(24, 20, 24, 20)
        self._root_layout.setSpacing(14)

        self.setStyleSheet(APP_STYLESHEET)

        # Stack
        self.stack = QStackedWidget(self)
        self._root_layout.addWidget(self.stack)

        self._build_intro_page()
        self._build_menu_page()
        self._build_upload_viewer_page()
        self._build_realtime_viewer_page()
        self._build_camera_viewer_page()

        self.stack.setCurrentWidget(self.intro_page)
        self._start_intro()

    # ── Page builders ─────────────────────────────────────────────────────────

    def _build_intro_page(self):
        self.intro_page = QWidget()
        self.intro_page.setObjectName("IntroPage")
        layout = QVBoxLayout(self.intro_page)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(20, 20, 20, 20)

        if INTRO_LOGO_PATH.exists():
            self._intro_widget = IntroLogoWidget(INTRO_LOGO_PATH, self)
            self._intro_widget.finished.connect(self._on_intro_finished)
            layout.addWidget(self._intro_widget, 0, Qt.AlignmentFlag.AlignCenter)
        else:
            self._intro_widget = IntroLogoWidget(INTRO_LOGO_PATH, self)
            self._intro_widget.finished.connect(self._on_intro_finished)
            layout.addWidget(self._intro_widget, 0, Qt.AlignmentFlag.AlignCenter)

        self.stack.addWidget(self.intro_page)

    def _build_menu_page(self):
        self.menu_page = QWidget()
        self.menu_page.setObjectName("MenuPage")
        layout = QVBoxLayout(self.menu_page)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(14)

        brand = QLabel("Program Skripsi", self)
        brand.setObjectName("BrandLabel")
        brand.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(brand)

        title = QLabel("Realtime Deepfake Detector (Xception)", self)
        title.setObjectName("TitleLabel")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        sub = QLabel(
            "Upload gambar/video untuk analisis statis, atau pantau layar/kamera secara real-time\n"
            "Deteksi menggunakan arsitektur Xception murni untuk klasifikasi wajah REAL/DEEPFAKE.",
            self,
        )
        sub.setObjectName("SubtitleLabel")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub.setWordWrap(True)
        layout.addWidget(sub)

        divider = QFrame(self)
        divider.setObjectName("Divider")
        divider.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(divider)

        upload_btn = QPushButton("   ↑   Upload Gambar / Video", self)
        upload_btn.setObjectName("PrimaryButton")
        upload_btn.setMinimumHeight(46)
        upload_btn.clicked.connect(self.open_upload_mode)
        layout.addWidget(upload_btn)

        realtime_btn = QPushButton("   ◉   Realtime Screening", self)
        realtime_btn.setObjectName("SecondaryButton")
        realtime_btn.setMinimumHeight(46)
        realtime_btn.clicked.connect(self.open_realtime_mode)
        layout.addWidget(realtime_btn)

        camera_btn = QPushButton("   📷   Camera Screening", self)
        camera_btn.setObjectName("SecondaryButton")
        camera_btn.setMinimumHeight(46)
        camera_btn.clicked.connect(self.open_camera_mode)
        layout.addWidget(camera_btn)

        self.stack.addWidget(self.menu_page)

    def _build_upload_viewer_page(self):
        self.upload_page = QWidget()
        self.upload_page.setObjectName("ViewerPage")
        outer = QVBoxLayout(self.upload_page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(12)

        header = QHBoxLayout()
        self._upload_title = QLabel("Upload Gambar / Video", self)
        self._upload_title.setObjectName("SectionTitle")
        icon = QLabel("↑", self)
        icon.setObjectName("SectionTitle")
        header.addWidget(icon)
        header.addWidget(self._upload_title)
        header.addStretch()
        self._upload_status = QLabel("Ready", self)
        self._upload_status.setObjectName("StatusPill")
        header.addWidget(self._upload_status)
        outer.addLayout(header)

        body = QHBoxLayout()
        body.setSpacing(16)

        left_col = QVBoxLayout()
        left_col.setSpacing(10)

        self.drop_area = UploadDropArea(self)
        self.drop_area.file_selected.connect(self._on_file_selected)
        left_col.addWidget(self.drop_area)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self._detect_btn = QPushButton("   🔍   Detect Deepfake", self)
        self._detect_btn.setObjectName("PrimaryButton")
        self._detect_btn.setMinimumHeight(44)
        self._detect_btn.setEnabled(False)
        self._detect_btn.clicked.connect(self._run_upload_detection)
        btn_row.addWidget(self._detect_btn, 1)

        self._export_btn = QPushButton("   💾   Export Video", self)
        self._export_btn.setObjectName("SecondaryButton")
        self._export_btn.setMinimumHeight(44)
        self._export_btn.setEnabled(False)
        self._export_btn.hide()
        self._export_btn.clicked.connect(self._run_video_export)
        btn_row.addWidget(self._export_btn, 1)

        self._clear_btn = QPushButton("🗑", self)
        self._clear_btn.setObjectName("DangerOutlineButton")
        self._clear_btn.setMinimumHeight(44)
        self._clear_btn.setFixedWidth(46)
        self._clear_btn.clicked.connect(self._clear_upload)
        btn_row.addWidget(self._clear_btn)

        left_col.addLayout(btn_row)

        hint = QLabel("🛡 File diproses lokal menggunakan model Xception .keras", self)
        hint.setObjectName("UploadSubText")
        left_col.addWidget(hint)

        left_col.addStretch()
        body.addLayout(left_col, 1)

        right_col = QVBoxLayout()
        right_col.setSpacing(10)

        hasil_title = QLabel("Hasil Deteksi", self)
        hasil_title.setObjectName("SectionTitle")
        icon2 = QLabel("📊", self)
        icon2.setObjectName("SectionTitle")

        title_row = QHBoxLayout()
        title_row.addWidget(icon2)
        title_row.addWidget(hasil_title)
        title_row.addStretch()
        right_col.addLayout(title_row)

        self._upload_image_label = QLabel(self)
        self._upload_image_label.setObjectName("PreviewSurface")
        self._upload_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._upload_image_label.setMinimumSize(380, 260)
        self._upload_image_label.setText("Gambar hasil deteksi muncul di sini")
        right_col.addWidget(self._upload_image_label, 1)

        self._result_scroll = QScrollArea(self)
        self._result_scroll.setWidgetResizable(True)
        self._result_scroll.setMaximumHeight(200)
        self._result_panel = DetectionResultPanel(self)
        self._result_scroll.setWidget(self._result_panel)
        self._result_scroll.hide()
        right_col.addWidget(self._result_scroll)

        body.addLayout(right_col, 1)
        outer.addLayout(body, 1)

        back_btn = QPushButton("← Kembali ke Menu", self)
        back_btn.setObjectName("GhostButton")
        back_btn.setMinimumHeight(38)
        back_btn.clicked.connect(self.back_to_menu)
        outer.addWidget(back_btn)

        self.stack.addWidget(self.upload_page)

    def _build_realtime_viewer_page(self):
        self.realtime_page = QWidget()
        self.realtime_page.setObjectName("ViewerPage")
        layout = QVBoxLayout(self.realtime_page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        header = QHBoxLayout()
        rt_title = QLabel("Realtime Screening", self)
        rt_title.setObjectName("ViewerTitle")
        header.addWidget(rt_title)
        header.addStretch()
        self._rt_status = QLabel("Idle", self)
        self._rt_status.setObjectName("StatusPill")
        header.addWidget(self._rt_status)
        layout.addLayout(header)

        self._rt_image_label = QLabel(self)
        self._rt_image_label.setObjectName("PreviewSurface")
        self._rt_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._rt_image_label.setMinimumSize(540, 360)
        self._rt_image_label.setText("Realtime preview")
        layout.addWidget(self._rt_image_label, 1)

        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(10)

        back_btn = QPushButton("← Kembali", self)
        back_btn.setObjectName("GhostButton")
        back_btn.setMinimumHeight(40)
        back_btn.clicked.connect(self.back_to_menu)
        ctrl_row.addWidget(back_btn)

        self._toggle_btn = QPushButton("▶   Start Screening", self)
        self._toggle_btn.setObjectName("PrimaryButton")
        self._toggle_btn.setMinimumHeight(40)
        self._toggle_btn.clicked.connect(self._toggle_screening)
        ctrl_row.addWidget(self._toggle_btn, 1)

        fps_lbl = QLabel("FPS", self)
        fps_lbl.setObjectName("ControlLabel")
        ctrl_row.addWidget(fps_lbl)

        self._fps_spinbox = QSpinBox(self)
        self._fps_spinbox.setObjectName("FpsSpinBox")
        self._fps_spinbox.setRange(5, 60)
        self._fps_spinbox.setValue(30)
        self._fps_spinbox.setSuffix(" fps")
        self._fps_spinbox.setMinimumHeight(40)
        self._fps_spinbox.setFixedWidth(100)
        ctrl_row.addWidget(self._fps_spinbox)

        layout.addLayout(ctrl_row)
        self.stack.addWidget(self.realtime_page)

    def _build_camera_viewer_page(self):
        self.camera_page = QWidget()
        self.camera_page.setObjectName("ViewerPage")
        layout = QVBoxLayout(self.camera_page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        header = QHBoxLayout()
        cam_title = QLabel("Camera Screening", self)
        cam_title.setObjectName("ViewerTitle")
        header.addWidget(cam_title)
        header.addStretch()
        self._cam_status = QLabel("Idle", self)
        self._cam_status.setObjectName("StatusPill")
        header.addWidget(self._cam_status)
        layout.addLayout(header)

        self._cam_image_label = QLabel(self)
        self._cam_image_label.setObjectName("PreviewSurface")
        self._cam_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._cam_image_label.setMinimumSize(540, 360)
        self._cam_image_label.setText("Camera preview")
        layout.addWidget(self._cam_image_label, 1)

        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(10)

        back_btn = QPushButton("← Kembali", self)
        back_btn.setObjectName("GhostButton")
        back_btn.setMinimumHeight(40)
        back_btn.clicked.connect(self.back_to_menu)
        ctrl_row.addWidget(back_btn)

        self._cam_toggle_btn = QPushButton("⏹   Stop Camera", self)
        self._cam_toggle_btn.setObjectName("PrimaryButton")
        self._cam_toggle_btn.setMinimumHeight(40)
        self._cam_toggle_btn.clicked.connect(self._toggle_camera)
        ctrl_row.addWidget(self._cam_toggle_btn, 1)

        layout.addLayout(ctrl_row)
        self.stack.addWidget(self.camera_page)

    def _start_intro(self):
        self._intro_widget.start()

    def _on_intro_finished(self):
        self.stack.setCurrentWidget(self.menu_page)

    # ── Navigation ────────────────────────────────────────────────────────────

    def open_upload_mode(self):
        self._stop_screening_if_needed()
        self.current_mode = "upload"
        self._pending_image_path = None
        self.drop_area.clear_preview()
        self._upload_image_label.clear()
        self._upload_image_label.setText("Gambar hasil deteksi muncul di sini")
        self._upload_status.setText("Ready")
        self._detect_btn.setEnabled(False)
        self._result_scroll.hide()
        self._result_panel.clear()
        self.stack.setCurrentWidget(self.upload_page)

    def open_realtime_mode(self):
        self._stop_screening_if_needed()
        self.current_mode = "realtime"
        self._rt_image_label.clear()
        self._rt_image_label.setText("Realtime preview")
        self._rt_status.setText("Idle")
        self._toggle_btn.setText("▶   Start Screening")
        self.stack.setCurrentWidget(self.realtime_page)

    def open_camera_mode(self):
        self._stop_screening_if_needed()
        self.current_mode = "camera"
        self._cam_image_label.clear()
        self._cam_image_label.setText("Connecting to camera...")
        self._cam_status.setText("Connecting")
        self.stack.setCurrentWidget(self.camera_page)
        # Langsung jalankan kamera
        QTimer.singleShot(100, self._toggle_camera)

    def back_to_menu(self):
        self._stop_screening_if_needed()
        self.current_mode = None
        self._rt_status.setText("Idle")
        self._cam_status.setText("Idle")
        self.stack.setCurrentWidget(self.menu_page)

    # ── Upload mode logic ──────────────────────────────────────────────────────

    def _on_file_selected(self, path: str):
        self._pending_image_path = path
        self.drop_area.set_preview(path)
        self._detect_btn.setEnabled(True)
        
        suffix = Path(path).suffix.lower()
        is_video = suffix in {".mp4", ".avi", ".mov", ".mkv"}
        if is_video:
            self._export_btn.show()
            self._export_btn.setEnabled(True)
        else:
            self._export_btn.hide()
            self._export_btn.setEnabled(False)

        self._upload_status.setText("Siap dideteksi")
        self._result_scroll.hide()
        self._result_panel.clear()
        self._upload_image_label.clear()
        self._upload_image_label.setText("Gambar hasil deteksi muncul di sini")

    def _clear_upload(self):
        self._pending_image_path = None
        self.drop_area.clear_preview()
        self._detect_btn.setEnabled(False)
        self._export_btn.hide()
        self._export_btn.setEnabled(False)
        self._upload_status.setText("Ready")
        self._result_scroll.hide()
        self._result_panel.clear()
        self._upload_image_label.clear()
        self._upload_image_label.setText("Gambar hasil deteksi muncul di sini")

    def _run_upload_detection(self):
        if not self._pending_image_path:
            return

        suffix = Path(self._pending_image_path).suffix.lower()
        is_video = suffix in {".mp4", ".avi", ".mov", ".mkv"}

        self._upload_status.setText("Memproses…")
        self._detect_btn.setEnabled(False)
        self._export_btn.setEnabled(False)
        self._clear_btn.setEnabled(False)
        QApplication.processEvents()

        if is_video:
            self._run_video_detection()
        else:
            self._run_image_detection()

    def _run_video_export(self):
        if not self._pending_image_path:
            return

        from PyQt6.QtWidgets import QFileDialog
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Simpan Hasil Deteksi Video",
            f"result_{Path(self._pending_image_path).stem}.mp4",
            "Video Files (*.mp4)"
        )
        
        if not save_path:
            return

        self._upload_status.setText("Exporting…")
        self._detect_btn.setEnabled(False)
        self._export_btn.setEnabled(False)
        self._clear_btn.setEnabled(False)
        QApplication.processEvents()

        self._run_video_detection(output_path=save_path)

    def _run_image_detection(self):
        try:
            self._ensure_models_loaded()
            file_bytes = np.fromfile(self._pending_image_path, dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError("File gambar tidak bisa dibaca.")

            # Menggunakan model Xception langsung
            annotated, detections = annotate_frame(
                frame,
                self.face_detector,
                self.deepfake_classifier,
                self.tf,
                self.input_size,
                return_detections=True,
                mode="upload"
            )

            self._upload_status.setText("Analyzed")
            self._update_upload_image(annotated)
            self._result_panel.show_results(detections)
            self._result_scroll.show()

        except Exception as exc:
            self._upload_status.setText("Error")
            QMessageBox.critical(self, "Model Error", str(exc))
        finally:
            self._detect_btn.setEnabled(True)
            self._clear_btn.setEnabled(True)

    def _run_video_detection(self, output_path=None):
        self._ensure_models_loaded()
        self.video_thread = VideoProcessorThread(
            self._pending_image_path, 
            output_path=output_path,
            face_detector=self.face_detector,
            feature_extractor=self.deepfake_classifier
        )
        self.video_thread.change_pixmap_signal.connect(self._update_upload_image)
        self.video_thread.progress_signal.connect(self._on_video_progress)
        self.video_thread.finished_signal.connect(self._on_video_finished)
        self.video_thread.error_signal.connect(self._handle_video_error)
        self.video_thread.start()

    def _on_video_progress(self, progress: int):
        status_text = "Exporting" if (self.video_thread and self.video_thread.output_path) else "Memproses"
        self._upload_status.setText(f"{status_text} {progress}%")

    def _on_video_finished(self, detections: list):
        self._upload_status.setText("Analyzed")
        self._result_panel.show_results(detections)
        self._result_scroll.show()
        self._detect_btn.setEnabled(True)
        self._export_btn.setEnabled(True)
        self._clear_btn.setEnabled(True)
        
        if self.video_thread and self.video_thread.output_path:
            QMessageBox.information(self, "Export Berhasil", f"Video hasil deteksi disimpan ke:\n{self.video_thread.output_path}")

        self.video_thread = None

    def _handle_video_error(self, message: str):
        self._upload_status.setText("Error")
        QMessageBox.critical(self, "Video Error", message)
        self._detect_btn.setEnabled(True)
        self._export_btn.setEnabled(True)
        self._clear_btn.setEnabled(True)
        self.video_thread = None

    def _update_upload_image(self, cv_img):
        h, w, ch = cv_img.shape
        qt_image = QImage(cv_img.data, w, h, ch * w, QImage.Format.Format_BGR888)
        scaled = qt_image.scaled(
            self._upload_image_label.width(),
            self._upload_image_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
        )
        self._upload_image_label.setPixmap(QPixmap.fromImage(scaled))

    # ── Realtime mode logic ────────────────────────────────────────────────────

    def _toggle_screening(self):
        if self.thread is not None and self.thread.isRunning():
            self._stop_screening_if_needed()
            self._rt_image_label.clear()
            self._rt_image_label.setText("Realtime preview")
            return

        if not self._enable_capture_exclusion():
            QMessageBox.warning(
                self,
                "Screen Capture",
                "Windows tidak mengizinkan aplikasi ini dikecualikan dari screen capture.\n"
                "Pindahkan jendela aplikasi dari area yang ingin dideteksi.",
            )

        self._ensure_models_loaded()
        self._toggle_btn.setText("⏹   Stop Screening")
        self._rt_status.setText("Screening")
        
        self.thread = ScreenProcessorThread(
            face_detector=self.face_detector,
            feature_extractor=self.deepfake_classifier
        )
        self.thread.target_fps = self._fps_spinbox.value()
        self.thread.change_pixmap_signal.connect(self._update_realtime_image)
        self.thread.error_signal.connect(self._handle_thread_error)
        self.thread.start()

    def _update_realtime_image(self, cv_img):
        h, w, ch = cv_img.shape
        qt_image = QImage(cv_img.data, w, h, ch * w, QImage.Format.Format_BGR888)
        scaled = qt_image.scaled(
            self._rt_image_label.width(),
            self._rt_image_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
        )
        self._rt_image_label.setPixmap(QPixmap.fromImage(scaled))

    # ── Camera mode logic ──────────────────────────────────────────────────────

    def _toggle_camera(self):
        if self.camera_thread is not None and self.camera_thread.isRunning():
            self._stop_screening_if_needed()
            self._cam_image_label.clear()
            self._cam_image_label.setText("Camera preview")
            return

        self._ensure_models_loaded()
        self._cam_toggle_btn.setText("⏹   Stop Camera")
        self._cam_status.setText("Streaming")
        
        self.camera_thread = CameraProcessorThread(
            camera_index=0,
            face_detector=self.face_detector,
            feature_extractor=self.deepfake_classifier
        )
        self.camera_thread.change_pixmap_signal.connect(self._update_camera_image)
        self.camera_thread.error_signal.connect(self._handle_thread_error)
        self.camera_thread.start()

    def _update_camera_image(self, cv_img):
        h, w, ch = cv_img.shape
        qt_image = QImage(cv_img.data, w, h, ch * w, QImage.Format.Format_BGR888)
        scaled = qt_image.scaled(
            self._cam_image_label.width(),
            self._cam_image_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
        )
        self._cam_image_label.setPixmap(QPixmap.fromImage(scaled))

    def _handle_thread_error(self, message: str):
        self._stop_screening_if_needed()
        self._rt_image_label.clear()
        self._rt_image_label.setText("Realtime preview")
        QMessageBox.critical(self, "Model Error", message)

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _ensure_models_loaded(self):
        """Memuat semua model, kini menggunakan Xception murni tanpa SVM."""
        if (self.face_detector is not None and 
            self.deepfake_classifier is not None and 
            self.tf is not None):
            return
            
        validate_face_model_environment()
        classifier_path = resolve_classifier_path()
        self.yolo_device = require_torch_gpu()
        
        self.tf = load_tensorflow()
        self.tf_device = configure_tensorflow_device(self.tf)
        
        # 1. Load YOLO Face Detector
        from ultralytics import YOLO
        self.face_detector = YOLO(str(FACE_MODEL_PATH))
        self.face_detector.to(self.yolo_device)
        
        # 2. Load Full Xception Classifier (.keras)
        print("Loading Xception Classifier Model...")
        self.deepfake_classifier = load_classifier_model(classifier_path, self.tf)
        print("Semua komponen model (YOLO + Xception) berhasil dimuat!")

    def _stop_screening_if_needed(self):
        if self.thread is not None and self.thread.isRunning():
            self.thread.stop()
        self.thread = None

        if self.video_thread is not None and self.video_thread.isRunning():
            self.video_thread.stop()
        self.video_thread = None

        if self.camera_thread is not None and self.camera_thread.isRunning():
            self.camera_thread.stop()
        self.camera_thread = None

        self._disable_capture_exclusion()
        self._toggle_btn.setText("▶   Start Screening")
        self._rt_status.setText("Idle")
        self._cam_toggle_btn.setText("▶   Start Camera")
        self._cam_status.setText("Idle")

    def _enable_capture_exclusion(self) -> bool:
        if self.capture_exclusion_enabled:
            return True
        enabled = set_window_excluded_from_capture(self, True)
        self.capture_exclusion_enabled = enabled
        return enabled

    def _disable_capture_exclusion(self):
        if not self.capture_exclusion_enabled:
            return
        set_window_excluded_from_capture(self, False)
        self.capture_exclusion_enabled = False

    def closeEvent(self, event):
        self._stop_screening_if_needed()
        if hasattr(self._intro_widget, "_stop"):
            self._intro_widget._stop()
        elif hasattr(self._intro_widget, "stop"):
            self._intro_widget.stop()
        event.accept()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This script runs with Python 3.12.7
# Part of the 3D Cell Analysis project

import sys, os
from turtle import color
import cv2
import numpy as np
import tifffile
import vtk
from scipy.ndimage import binary_closing, binary_fill_holes, generate_binary_structure
from sklearn.ensemble import RandomForestClassifier

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QToolButton, QLabel, QPushButton, QAction, QFileDialog, QSlider,
    QStatusBar, QSpacerItem, QSizePolicy, QMessageBox, QStackedWidget,
    QDialog, QDialogButtonBox, QMenu, QProgressDialog, QFormLayout, QDoubleSpinBox, QInputDialog
)

from PyQt5.QtCore import Qt, QCoreApplication, QSize
from PyQt5.QtGui import (
    QImage, QPixmap, QPainter, QPen, QPolygonF, QColor,
    QMovie
)

from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkFiltersCore import vtkCleanPolyData

# ------------------------------------------------------------
# Helper function to convert to grayscale
# ------------------------------------------------------------
def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] in (3, 4):
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image

# ------------------------------------------------------------
# Freeform ROI widget
# ------------------------------------------------------------
class FreeformImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.freeform_mode = False
        self.drawing = False
        self.points: list = []
        self.freeform_polygon = None
        self.roi_selected_callback = None
        self.update_pixel_callback = None

    def start_freeform(self) -> None:
        self.freeform_mode = True
        self.drawing = False
        self.points.clear()
        self.freeform_polygon = None
        self.update()

    def paintEvent(self, a0) -> None:
        super().paintEvent(a0)
        if self.freeform_mode and self.points:
            painter = QPainter(self)
            painter.setPen(QPen(QColor('red'), 2, Qt.PenStyle.SolidLine))
            painter.drawPolygon(QPolygonF(self.points))

    def mousePressEvent(self, a0) -> None:
        if self.freeform_mode:
            self.points = [a0.pos()]
            self.drawing = True
            self.update()
        else:
            super().mousePressEvent(a0)

    def mouseMoveEvent(self, a0) -> None:
        if self.freeform_mode and self.drawing:
            self.points.append(a0.pos())
            self.update()
        if self.update_pixel_callback:
            self.update_pixel_callback(a0)
        super().mouseMoveEvent(a0)

    def mouseReleaseEvent(self, a0) -> None:
        if self.freeform_mode and self.drawing:
            self.points.append(a0.pos())
            self.drawing = False
            self.freeform_polygon = QPolygonF(self.points)
            self.freeform_mode = False
            self.update()
            if self.roi_selected_callback:
                self.roi_selected_callback(self.freeform_polygon)
        else:
            super().mouseReleaseEvent(a0)

# ------------------------------------------------------------
# Brush widget for ML annotation
# ------------------------------------------------------------
class BrushImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.drawing = False
        self.brush_size = 10
        self.mode = 1  # 1=signal, 2=background
        self.annotation_mask: np.ndarray = None
        self.image: QImage = None

    def setImage(self, qimage: QImage) -> None:
        self.image = qimage.copy()
        self.annotation_mask = np.zeros((qimage.height(), qimage.width()), np.uint8)
        self.setPixmap(QPixmap.fromImage(self.image))

    def setBrushSize(self, size: int) -> None:
        self.brush_size = size

    def setMode(self, mode: int) -> None:
        self.mode = mode

    def mousePressEvent(self, a0) -> None:
        self.drawing = True
        self.drawBrush(a0.pos())

    def mouseMoveEvent(self, a0) -> None:
        if self.drawing:
            self.drawBrush(a0.pos())

    def mouseReleaseEvent(self, a0) -> None:
        self.drawing = False

    def drawBrush(self, pos) -> None:
        if self.image is None:
            return
        painter = QPainter(self.image)
        color = QColor(0, 255, 0) if self.mode == 1 else QColor(255, 0, 255)
        painter.setPen(QPen(
            color,
            self.brush_size,
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin
        ))
        painter.drawPoint(pos)
        painter.end()
        x, y = pos.x(), pos.y()
        cv2.circle(self.annotation_mask, (x, y), self.brush_size // 2, int(self.mode), -1)
        self.setPixmap(QPixmap.fromImage(self.image))

# ------------------------------------------------------------
# Threshold ROI selection dialog
# ------------------------------------------------------------
class ThresholdROIDialog(QDialog):
    def __init__(self, image_volume, initial_frame=0, initial_roi_results=None, colormap= cv2.COLORMAP_JET, parent=None):
        super().__init__(parent)
        self.colormap = colormap
        self.setWindowTitle("Threshold ROI Selector")
        if image_volume.ndim == 2:
            self.image_volume = image_volume[np.newaxis, ...]
        elif image_volume.ndim == 3 and image_volume.shape[2] in (3, 4):
            gray = cv2.cvtColor(image_volume, cv2.COLOR_RGB2GRAY)
            self.image_volume = gray[np.newaxis, ...]
        elif image_volume.ndim == 3:
            self.image_volume = image_volume
        elif image_volume.ndim == 4:
            frames = []
            for f in image_volume:
                frames.append(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) if f.ndim == 3 else f)
            self.image_volume = np.array(frames)
        else:
            self.image_volume = image_volume[np.newaxis, ...]

        self.num_frames = self.image_volume.shape[0]
        self.current_frame = initial_frame
        self.roi_results = initial_roi_results.copy() if initial_roi_results else {}

        layout = QVBoxLayout(self)
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.preview_label)
        self.info_label = QLabel()
        layout.addWidget(self.info_label)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 255)
        self.slider.setValue(128)
        self.slider.valueChanged.connect(self.update_preview)
        layout.addWidget(self.slider)

        btns = QHBoxLayout()
        for text, fn in [
            ("Back 10", self.go_back),
            ("Forward 10", self.go_forward),
            ("Apply", self.apply_current_roi),
            ("Clear", self.clear_rois),
            ("Render", self.accept),
            ("Cancel", self.reject),
        ]:
            b = QPushButton(text)
            b.clicked.connect(fn)
            btns.addWidget(b)
        layout.addLayout(btns)

        self.update_preview()

    def compute_contour(self):
        frame = self.image_volume[self.current_frame]
        _, bin_ = cv2.threshold(frame, self.slider.value(), 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(bin_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return (max(cnts, key=cv2.contourArea), bin_) if cnts else (None, bin_)

    def update_preview(self) -> None:
        cnt, _ = self.compute_contour()
        gray = self.image_volume[self.current_frame]
        img = cv2.applyColorMap(gray, self.colormap)
        if cnt is not None:
            cv2.drawContours(img, [cnt], -1, (0.0, 255.0, 255.0), 2)
        h, w, _ = img.shape
        qt = QImage(img.data, w, h, 3*w, QImage.Format_RGB888).rgbSwapped()
        pix = QPixmap.fromImage(qt).scaled(
            300, 300,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.preview_label.setPixmap(pix)
        self.info_label.setText(f"Frame: {self.current_frame+1}/{self.num_frames}")

    def go_back(self) -> None:
        self.current_frame = max(0, self.current_frame - 10)
        self.update_preview()

    def go_forward(self) -> None:
        self.current_frame = min(self.num_frames-1, self.current_frame + 10)
        self.update_preview()

    def apply_current_roi(self) -> None:
        cnt, _ = self.compute_contour()
        if cnt is None:
            QMessageBox.warning(self, "No ROI", "No contours found.")
            return
        mask = np.zeros_like(self.image_volume[self.current_frame], np.uint8)
        cv2.drawContours(mask, [cnt], -1, (255.0,), -1)
        self.roi_results[self.current_frame] = {'mask': mask, 'contour': cnt}
        self.update_preview()

    def clear_rois(self) -> None:
        if QMessageBox.question(self, "Clear ROIs", "Clear all ROIs?",
                                QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            self.roi_results.clear()
            self.update_preview()

# ------------------------------------------------------------
# ML Pixel Classifier dialog
# ------------------------------------------------------------
class MLPixelClassifierDialog(QDialog):
    def __init__(self, image: np.ndarray, colormap=cv2.COLORMAP_JET, parent=None):
        super().__init__(parent)
        self.colormap = colormap
        self.setWindowTitle("ML Pixel Classifier")
        self.image = image

        # Colorize the background with the chosen LUT
        colored = cv2.applyColorMap(self.image, self.colormap)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        h, w, _ = colored.shape
        qimg = QImage(colored.data, w, h, 3*w, QImage.Format_RGB888)

        # Set up the brush canvas
        self.brush_label = BrushImageLabel()
        self.brush_label.setImage(qimg)

        # Controls
        self.signal_btn = QPushButton("Signal")
        self.signal_btn.clicked.connect(lambda: self.brush_label.setMode(1))
        self.bg_btn = QPushButton("Background")
        self.bg_btn.clicked.connect(lambda: self.brush_label.setMode(2))
        self.train_btn = QPushButton("Train & Apply")
        self.train_btn.clicked.connect(self.train_and_apply)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(1, 50)
        slider.setValue(10)
        slider.valueChanged.connect(lambda v: self.brush_label.setBrushSize(v))

        # Layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.brush_label)
        hl = QHBoxLayout()
        for wgt in (self.signal_btn, self.bg_btn, self.train_btn, self.cancel_btn):
            hl.addWidget(wgt)
        layout.addLayout(hl)
        layout.addWidget(QLabel("Brush Size"))
        layout.addWidget(slider)

        # Placeholders for later
        self.training_data = None
        self.training_labels = None
        self.result_segmentation = None

    def train_and_apply(self) -> None:
        ann = self.brush_label.annotation_mask
        sig = np.where(ann == 1)
        bg = np.where(ann == 2)
        if not len(sig[0]) or not len(bg[0]):
            QMessageBox.warning(self, "Insufficient Data", "Annotate both signal and background.")
            return
        X, y = [], []
        for i, j in zip(*sig):
            X.append([float(self.image[i, j])]); y.append(1)
        for i, j in zip(*bg):
            X.append([float(self.image[i, j])]); y.append(0)
        X = np.array(X); y = np.array(y)
        clf = RandomForestClassifier(n_estimators=50)
        clf.fit(X, y)
        self.training_data = X
        self.training_labels = y

        h, w = self.image.shape
        pred = clf.predict(self.image.reshape(-1, 1).astype(np.float32)).reshape(h, w)
        seg = np.zeros((h, w, 3), np.uint8)
        seg[pred == 1] = [0, 255, 0]
        seg[pred == 0] = [255, 0, 255]
        self.result_segmentation = seg

        q = QImage(seg.data, w, h, 3*w, QImage.Format_RGB888)
        self.brush_label.setPixmap(QPixmap.fromImage(q))
        self.accept()

# ------------------------------------------------------------
# Surface filter dialog
# ------------------------------------------------------------
class FilterSurfacesDialog(QDialog):
    def __init__(self, polydata: vtk.vtkPolyData, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Filter Surfaces by Size")
        self.polydata = polydata

        self.low_slider = QSlider(Qt.Orientation.Horizontal)
        self.low_slider.setRange(0, 1000)
        self.low_slider.valueChanged.connect(self.update_preview)
        self.high_slider = QSlider(Qt.Orientation.Horizontal)
        self.high_slider.setRange(0, 1000)
        self.high_slider.setValue(1000)
        self.high_slider.valueChanged.connect(self.update_preview)

        self.low_lbl = QLabel("Lower: 0")
        self.high_lbl = QLabel("Upper:1000")

        self.vtk_widget = QVTKRenderWindowInteractor(self)
        self.vtk_renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.vtk_renderer)
        self.vtk_widget.Initialize()
        self.vtk_widget.Start()

        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(self.apply_filter)
        box.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        slayout = QHBoxLayout()
        slayout.addWidget(self.low_lbl)
        slayout.addWidget(self.low_slider)
        slayout.addWidget(self.high_lbl)
        slayout.addWidget(self.high_slider)
        layout.addLayout(slayout)
        layout.addWidget(self.vtk_widget)
        layout.addWidget(box)

        self.filtered_polydata = None
        self.update_preview()

    def update_preview(self) -> None:
        lower = self.low_slider.value()
        upper = self.high_slider.value()
        self.low_lbl.setText(f"Lower: {lower}")
        self.high_lbl.setText(f"Upper: {upper}")

        # ─── STEP 1: Clean the mesh ────────────────────────────────────
        cleaner = vtkCleanPolyData()
        cleaner.SetInputData(self.polydata)
        cleaner.Update()
        cleaned = cleaner.GetOutput()

        # ─── STEP 2: Do connectivity on the *cleaned* mesh ───────────
        conn = vtk.vtkPolyDataConnectivityFilter()
        conn.SetInputData(cleaned)
        conn.SetExtractionModeToAllRegions()
        conn.Update()
        n = conn.GetNumberOfExtractedRegions()

        prog = QProgressDialog("Filtering surfaces...", "Cancel", 0, n, self)
        prog.setWindowModality(Qt.WindowModal)

        app = QCoreApplication.instance()
        af = vtk.vtkAppendPolyData()
        for rid in range(n):
            prog.setValue(rid)
            app.processEvents()
            if prog.wasCanceled():
                break

            rc = vtk.vtkPolyDataConnectivityFilter()
            rc.SetInputData(cleaned)
            rc.SetExtractionModeToSpecifiedRegions()
            rc.AddSpecifiedRegion(rid)
            rc.Update()
            poly = rc.GetOutput()

            mass = vtk.vtkMassProperties()
            mass.SetInputData(poly)
            mass.Update()
            vol = mass.GetVolume()
            if lower <= vol <= upper:
                af.AddInputData(poly)

        af.Update()
        out = af.GetOutput()
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(out)
        mapper.ScalarVisibilityOff()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1, 0, 0)

        self.vtk_renderer.RemoveAllViewProps()
        self.vtk_renderer.AddActor(actor)
        self.vtk_renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

        self.filtered_polydata = out

    def apply_filter(self) -> None:
        self.update_preview()
        self.accept()

# ----------------------------------------------------------------------------
# Voxel Settings dialog (PSF + µm‑scale entries)
# ----------------------------------------------------------------------------
class VoxelDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Voxel / PSF Settings")
        self.setMinimumWidth(300)

        layout = QVBoxLayout(self)

        # PSF image display (just load "psf.png" from your working folder)
        self.image_label = QLabel()
        psf_pix = QPixmap("psf.png")
        if not psf_pix.isNull():
            self.image_label.setPixmap(psf_pix.scaled(250, 250, Qt.KeepAspectRatio))
        else:
            self.image_label.setText("PSF image\nnot found")
            self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)

        # Form to enter X, Y, Z‑step scales (in µm)
        form = QFormLayout()
        self.xSpin = QDoubleSpinBox()
        self.xSpin.setRange(0.01, 1000.0)
        self.xSpin.setDecimals(3)
        self.xSpin.setSuffix(" µm")
        self.xSpin.setValue(getattr(parent, "voxel_x", 1.0))
        form.addRow("X scale:", self.xSpin)

        self.ySpin = QDoubleSpinBox()
        self.ySpin.setRange(0.01, 1000.0)
        self.ySpin.setDecimals(3)
        self.ySpin.setSuffix(" µm")
        self.ySpin.setValue(getattr(parent, "voxel_y", 1.0))
        form.addRow("Y scale:", self.ySpin)

        self.zSpin = QDoubleSpinBox()
        self.zSpin.setRange(0.01, 1000.0)
        self.zSpin.setDecimals(3)
        self.zSpin.setSuffix(" µm")
        self.zSpin.setValue(getattr(parent, "voxel_z", 1.0))
        form.addRow("Z‑step:", self.zSpin)

        layout.addLayout(form)

        # OK / Cancel buttons
        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        layout.addWidget(box)

# ------------------------------------------------------------
# Main Application
# ------------------------------------------------------------
class ImageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Cell Analysis")
        self.resize(1000, 700)

        # Data holders
        self.raw_image_data       = None
        self.corrected_image_data = None
        self.adjusted_image_data  = None
        self.roi_results          = {}
        self.num_frames           = 1
        self.current_frame_index  = 0
        self.brightness           = 0
        self.contrast             = 1.0
        self.current_view_mode    = "2D"
        self.show_raw             = True
        self.process_log          = []
        self.current_polydata     = None
        self.final_polydata       = None

        #current LUT
        self.lut_name = "JET"
        self.colormap = cv2.COLORMAP_JET

        # ML training
        self.ml_X_train       = None
        self.ml_y_train       = None
        self.ml_training_data = {}
        self.ml_classifier    = None

        # GPU support
        self.use_gpu          = False
    

        # Initialize the three voxel attributes and data holders
        self.voxel_x = 1.0
        self.voxel_y = 1.0
        self.voxel_z = 1.0

        # Build UI (without spinner)
        self.init_ui()
        self.gpu_supported = self.is_gpu_supported()

        # Menu and status bar
        self.create_menu()
        self._status_bar = QStatusBar(self)
        self.setStatusBar(self._status_bar)

        # ——————————————————————————
        # spinner setup (after status bar!)
        self.spinner_label = QLabel(self)
        self.spinner_movie = QMovie("spinner.gif")
        # scale the raw frames down to 24 x 24 px
        self.spinner_movie.setScaledSize(QSize(24, 24))
        self.spinner_label.setMovie(self.spinner_movie)
        # force the Qlabel to exactly 24 x 24 px
        self.spinner_label.setFixedSize(24, 24)
        self.spinner_label.hide()
        self.statusBar().addPermanentWidget(self.spinner_label)
        # ——————————————————————————

        self.set_status("Ready")

    def is_gpu_supported(self) -> bool:
        try:
            gm = vtk.vtkGPUVolumeRayCastMapper()
            return gm.IsRenderSupported(self.vtk_widget.GetRenderWindow()) == 1
        except:
            return False

    def set_status(self, msg: str, timeout: int = 0) -> None:
        ind = "GPU: ●" if self.gpu_supported else "GPU: X"
        self._status_bar.showMessage(f"{ind} | {msg}", timeout)

    def init_ui(self) -> None:
        # Top control row
        top = QHBoxLayout()
        top.addStretch()
        self.pixel_value_label = QLabel("Pixel: N/A")
        self.pixel_value_label.setStyleSheet("background-color: #eee; padding:2px;")
        top.addWidget(self.pixel_value_label)

        self.toggle_button = QPushButton("Switch to 3D View")
        self.toggle_button.clicked.connect(self.toggle_view)
        top.addWidget(self.toggle_button)

        self.toggle_raw_button = QPushButton("Show Corrected")
        self.toggle_raw_button.clicked.connect(self.toggle_raw_corrected)
        top.addWidget(self.toggle_raw_button)

        self.bg_corr_btn = QPushButton("Background Correction")
        self.bg_corr_btn.clicked.connect(self.start_background_correction)
        top.addWidget(self.bg_corr_btn)

        self.del_btn = QPushButton("Delete")
        self.del_btn.clicked.connect(self.start_delete)
        top.addWidget(self.del_btn)

        self.thresh_btn = QPushButton("Threshold ROI")
        self.thresh_btn.clicked.connect(self.on_threshold_roi)
        top.addWidget(self.thresh_btn)

        self.ml_btn = QPushButton("ML Pixel Classifier")
        self.ml_btn.clicked.connect(self.ml_pixel_classifier)
        top.addWidget(self.ml_btn)

        # ML stack dropdown
        self.ml_stack_button = QToolButton()
        self.ml_stack_button.setText("Classify Entire Stack")
        self.ml_stack_button.setPopupMode(QToolButton.MenuButtonPopup)
        self.ml_stack_menu = QMenu(self)
        act_comb = QAction("Combined", self)
        act_comb.triggered.connect(self.ml_classify_stack_combined)
        self.ml_stack_menu.addAction(act_comb)
        self.ml_slice_menu = QMenu("Slice", self)
        self.ml_stack_menu.addMenu(self.ml_slice_menu)
        self.ml_stack_button.setMenu(self.ml_stack_menu)
        top.addWidget(self.ml_stack_button)

        self.filter_btn = QPushButton("Filter Surfaces")
        self.filter_btn.clicked.connect(self.filter_surfaces)
        top.addWidget(self.filter_btn)

        self.del_surf_btn = QPushButton("Delete Surfaces")
        self.del_surf_btn.clicked.connect(self.delete_surfaces)
        top.addWidget(self.del_surf_btn)

        self.clean_surf_btn = QPushButton("Clean Surfaces")
        self.clean_surf_btn.clicked.connect(self.clean_surfaces)
        top.addWidget(self.clean_surf_btn)

        # 3D Options dropdown
        self.surface_menu = QMenu("3D Options", self)
        self.surface_menu.addAction("3D Surface", self.render_roi_surface)
        self.surface_menu.addAction("Final Structure", self.render_final_structure)
        self.surface_btn = QPushButton("3D Options")
        self.surface_btn.setMenu(self.surface_menu)
        top.addWidget(self.surface_btn)

        # Central area
        central_widget = QWidget()
        central_layout = QVBoxLayout(central_widget)
        central_layout.addLayout(top)

        self.display_stack = QStackedWidget()
        self.image_label   = FreeformImageLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.update_pixel_callback = self.handle_mouse_move
        self.display_stack.addWidget(self.image_label)

        self.vtk_widget    = QVTKRenderWindowInteractor(self)
        self.vtk_renderer  = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.vtk_renderer)
        self.vtk_widget.Initialize()
        self.vtk_widget.Start()
        self.display_stack.addWidget(self.vtk_widget)

        self.display_stack.setCurrentIndex(0)
        central_layout.addWidget(self.display_stack, stretch=1)

        # Z-slider
        zlay = QHBoxLayout()
        zlay.addWidget(QLabel("Z-Frame:"))
        self.z_frame_label = QLabel("Frame: 0/0")
        zlay.addWidget(self.z_frame_label)
        self.z_slider = QSlider(Qt.Orientation.Horizontal)
        self.z_slider.valueChanged.connect(self.change_frame)
        self.z_slider.hide()
        zlay.addWidget(self.z_slider)
        central_layout.addLayout(zlay)

        # Brightness/Contrast
        adj = QHBoxLayout()
        adj.addWidget(QLabel("Brightness"))
        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(self.adjust_image)
        adj.addWidget(self.brightness_slider)
        adj.addSpacerItem(QSpacerItem(20,20,QSizePolicy.Expanding,QSizePolicy.Minimum))
        adj.addWidget(QLabel("Contrast"))
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(10, 300)
        self.contrast_slider.setValue(100)
        self.contrast_slider.valueChanged.connect(self.adjust_image)
        adj.addWidget(self.contrast_slider)
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset_adjustments)
        adj.addWidget(self.reset_btn)
        self.auto_btn = QPushButton("Auto Adjust")
        self.auto_btn.clicked.connect(self.auto_adjust_image)
        adj.addWidget(self.auto_btn)

        # LUT selection
        self.lut_btn = QPushButton("LUT")
        self.lut_btn.clicked.connect(self.choose_lut)
        top.addWidget(self.lut_btn)
        central_layout.addLayout(adj)

        self.setCentralWidget(central_widget)

        # ——————————————————————————
        # spinner setup (bottom‑right of status bar)
        self.spinner_label = QLabel(self)
        self.spinner_movie = QMovie("spinner.gif")
        self.spinner_label.setMovie(self.spinner_movie)
        self.spinner_label.hide()
        self.statusBar().addPermanentWidget(self.spinner_label)
        # ——————————————————————————

    def create_menu(self) -> None:
        menubar = self.menuBar()
        # ——————————————————————————File menu
        filem = menubar.addMenu("File")
        oa = QAction("Open...", self); oa.triggered.connect(self.open_image); filem.addAction(oa)
        ca = QAction("Close Image", self);ca.triggered.connect(self.close_image); filem.addAction(ca)
        filem.addSeparator()
        sa = QAction("Save As...", self); sa.triggered.connect(self.save_image);  filem.addAction(sa)
        sla= QAction("Save Log", self);    sla.triggered.connect(self.save_log);   filem.addAction(sla)
        sri= QAction("Save ROIs", self);   sri.triggered.connect(self.save_rois);  filem.addAction(sri)
        s3d= QAction("Save 3D Surface", self); s3d.triggered.connect(self.save_3d_surface); filem.addAction(s3d)
        sfu= QAction("Save Final Structure", self); sfu.triggered.connect(self.save_final_structure); filem.addAction(sfu)
        filem.addSeparator()
        ga = QAction("Use GPU Acceleration", self); ga.setCheckable(True); ga.triggered.connect(self.toggle_gpu_acceleration)
        filem.addAction(ga); filem.addSeparator()
        ora= QAction("Open ROIs", self);          ora.triggered.connect(self.open_rois);           filem.addAction(ora)
        o3d= QAction("Open 3D Surface", self);    o3d.triggered.connect(self.open_3d_surface);   filem.addAction(o3d)
        ofu= QAction("Open Final Structure", self); ofu.triggered.connect(self.open_final_structure); filem.addAction(ofu)

        # ——————————————————————————Voxel Menu
        voxel_menu = menubar.addMenu("Voxel")
        voxel_act = QAction("PSF Settings…", self)
        voxel_act.triggered.connect(self.open_voxel_dialog)
        voxel_menu.addAction(voxel_act) 

    def open_voxel_dialog(self) -> None:
        dlg = VoxelDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            self.voxel_x = dlg.xSpin.value()
            self.voxel_y = dlg.ySpin.value()
            self.voxel_z = dlg.zSpin.value()
            self.add_log(
                f"Voxel scales set: X={self.voxel_x} µm, "
                f"Y={self.voxel_y} µm, Z‑step={self.voxel_z} µm"
            )
            QMessageBox.information(
                self,
                "Voxel Settings",
                f"Scales saved:\n X={self.voxel_x} µm\n Y={self.voxel_y} µm\n Z‑step={self.voxel_z} µm"
            )

    # --- File operations ---
    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if not path:
            return
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext in (".tif", ".tiff"):
                img = tifffile.imread(path)
            else:
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise ValueError("cv2 failed to load image")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image:\n{e}")
            return

        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Determine frame count
        if img.ndim == 2:
            self.num_frames = 1
        elif img.ndim == 3 and img.shape[2] in (3, 4):
            self.num_frames = 1
        elif img.ndim == 3:
            self.num_frames = img.shape[0]
        elif img.ndim == 4:
            self.num_frames = img.shape[0]
        else:
            self.num_frames = 1

        self.raw_image_data       = img.copy()
        self.corrected_image_data = img.copy()
        self.adjusted_image_data  = None
        self.current_frame_index  = 0
        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(100)

        if self.num_frames > 1:
            self.z_slider.setRange(0, self.num_frames - 1)
            self.z_slider.setValue(0)
            self.z_slider.show()
            self.z_frame_label.setText(f"Frame: 1/{self.num_frames}")
        else:
            self.z_slider.hide()
            self.z_frame_label.setText("Frame: 1/1")

        self.add_log(f"Opened image: {path}")
        self.adjust_image()

    def save_image(self) -> None:
        if self.adjusted_image_data is None:
            QMessageBox.warning(self, "No image", "Nothing to save.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Image As", "",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;TIFF (*.tif *.tiff)")
        if not path:
            return
        img = self.adjusted_image_data.copy()
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(path, img):
            QMessageBox.critical(self, "Error", "Failed to save image.")
        else:
            QMessageBox.information(self, "Saved", f"Image saved to {path}")
            self.add_log(f"Saved image: {path}")

    def save_log(self) -> None:
        if not self.process_log:
            QMessageBox.information(self, "Log", "No log entries.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Log", "", "Text (*.txt)")
        if not path:
            return
        try:
            with open(path, "w") as f:
                for e in self.process_log:
                    f.write(e + "\n")
            QMessageBox.information(self, "Saved", f"Log saved to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save log:\n{e}")

    def save_rois(self) -> None:
        if not self.roi_results:
            QMessageBox.warning(self, "No ROIs", "Nothing to save.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save ROIs", "", "NumPy (*.npy)")
        if not path:
            return
        try:
            np.save(path, self.roi_results)
            QMessageBox.information(self, "Saved", f"ROIs saved to {path}")
            self.add_log(f"Saved ROIs: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save ROIs:\n{e}")

    def save_3d_surface(self) -> None:
        if self.current_polydata is None:
            QMessageBox.warning(self, "No surface", "Nothing to save.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save 3D Surface", "", "STL (*.stl)")
        if not path:
            return
        writer = vtk.vtkSTLWriter()
        writer.SetFileName(path)
        writer.SetInputData(self.current_polydata)
        writer.Write()
        QMessageBox.information(self, "Saved", f"3D surface saved to {path}")
        self.add_log(f"Saved 3D surface: {path}")

    def save_final_structure(self) -> None:
        if self.final_polydata is None:
            QMessageBox.warning(self, "No structure", "Nothing to save.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Final Structure", "", "STL (*.stl)")
        if not path:
            return
        writer = vtk.vtkSTLWriter()
        writer.SetFileName(path)
        writer.SetInputData(self.final_polydata)
        writer.Write()
        QMessageBox.information(self, "Saved", f"Final structure saved to {path}")
        self.add_log(f"Saved final structure: {path}")

    def open_rois(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open ROIs", "", "NumPy (*.npy)")
        if not path:
            return
        try:
            rois = np.load(path, allow_pickle=True).item()
            self.roi_results = rois
            QMessageBox.information(self, "Opened", f"ROIs loaded from {path}")
            self.add_log(f"Opened ROIs: {path}")
            self.update_image_display()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open ROIs:\n{e}")

    def open_3d_surface(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open 3D Surface", "", "STL (*.stl)")
        if not path:
            return
        reader = vtk.vtkSTLReader()
        reader.SetFileName(path)
        reader.Update()
        poly = reader.GetOutput()
        self.current_polydata = poly
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)
        mapper.ScalarVisibilityOff()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1, 0.5, 0.5)
        self.vtk_renderer.RemoveAllViewProps()
        self.vtk_renderer.AddActor(actor)
        self.vtk_renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()
        QMessageBox.information(self, "Opened", f"3D Surface loaded from {path}")
        self.add_log(f"Opened 3D surface: {path}")

    def open_final_structure(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open Final Structure", "", "STL (*.stl)")
        if not path:
            return
        reader = vtk.vtkSTLReader()
        reader.SetFileName(path)
        reader.Update()
        poly = reader.GetOutput()
        self.final_polydata = poly
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)
        mapper.ScalarVisibilityOff()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1, 1, 0)
        self.vtk_renderer.RemoveAllViewProps()
        self.vtk_renderer.AddActor(actor)
        self.vtk_renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()
        QMessageBox.information(self, "Opened", f"Final structure loaded from {path}")
        self.add_log(f"Opened final structure: {path}")

    # ------------------------------------------------------------
    # Helper method for LUT selection
    # ------------------------------------------------------------
    def choose_lut(self):
        items = ["JET","HOT","PARULA","OCEAN","HSV"]
        cmap_map = {
            "JET":     cv2.COLORMAP_JET,
            "HOT":     cv2.COLORMAP_HOT,
            "PARULA":  cv2.COLORMAP_PARULA,
            "OCEAN":   cv2.COLORMAP_OCEAN,
            "HSV":     cv2.COLORMAP_HSV,
        }
        idx = items.index(self.lut_name)
        lut, ok = QInputDialog.getItem(self, "Select LUT", "Lookup table:", items, idx, False)
        if not ok: 
            return
        self.lut_name = lut
        self.colormap = cmap_map[lut]
        self.add_log(f"Colormap set to {lut}")
        # refresh both views
        self.update_image_display()
        if self.current_view_mode=="3D":
            self.update_3d_view()

    # ------------------------------------------------------------
    # Background correction & deletion via freeform ROI
    # ------------------------------------------------------------
    def start_background_correction(self) -> None:
        if self.adjusted_image_data is None and self.raw_image_data is None:
            QMessageBox.warning(self, "No image", "Load an image first.")
            return
        QMessageBox.information(self, "Freeform ROI", "Draw around the object; pixels outside will be background.")
        self.image_label.start_freeform()
        self.image_label.roi_selected_callback = self.on_freeform_roi_selected

    def on_freeform_roi_selected(self, poly: QPolygonF) -> None:
        # 1) figure out the 2D slice shape you actually displayed
        frame0 = self.get_current_frame(self.raw_image_data)
        h_img, w_img = frame0.shape[:2]

        # 2) map your widget‑coords polygon back to image coords
        h_disp = self.image_label.height()
        w_disp = self.image_label.width()
        scale = min(w_disp / w_img, h_disp / h_img)
        off_x = (w_disp - w_img * scale) / 2
        off_y = (h_disp - h_img * scale) / 2

        pts = []
        for p in poly:
            x = int((p.x() - off_x) / scale)
            y = int((p.y() - off_y) / scale)
            x = np.clip(x, 0, w_img - 1)
            y = np.clip(y, 0, h_img - 1)
            pts.append([x, y])

        # 3) build a mask the same shape as that slice
        mask = np.zeros((h_img, w_img), np.uint8)
        cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 1)

        # 4) subtract frame‑by‑frame using that same mask
        corrected_stack = []
        for i in range(self.num_frames):
            # grab the exact same slice shape from the raw data
            raw_frame = self.get_current_frame(
                self.raw_image_data if self.raw_image_data.ndim == 2 else self.raw_image_data[i]
            )
            gray = (
                cv2.cvtColor(raw_frame, cv2.COLOR_RGB2GRAY)
                if raw_frame.ndim == 3
                else raw_frame
            )
            # now mask==0 and gray have the same H×W
            bg = gray[mask == 0].mean()
            corr = np.clip(raw_frame.astype(np.int16) - int(bg), 0, 255).astype(np.uint8)
            corrected_stack.append(corr)

        # build corrected_image_data in the same shape as raw_image_data
        if self.num_frames == 1:
            self.corrected_image_data = corrected_stack[0]
        else:
            self.corrected_image_data = np.stack(corrected_stack, axis=0)

        self.add_log("Background corrected via freeform ROI")
        self.adjust_image()

        QMessageBox.information(self,
            "Background Correction",
            "Per‑frame background subtraction applied successfully."
        )
    def start_delete(self) -> None:
        if self.adjusted_image_data is None and self.raw_image_data is None:
            QMessageBox.warning(self, "No image", "Load an image first.")
            return
        QMessageBox.information(self, "Delete ROI", "Draw around pixels to delete.")
        self.image_label.start_freeform()
        self.image_label.roi_selected_callback = self.on_delete_roi_selected

    def on_delete_roi_selected(self, poly: QPolygonF) -> None:
        # same H×W logic
        frame0 = self.get_current_frame(self.raw_image_data)
        h_img, w_img = frame0.shape[:2]

        h_disp = self.image_label.height()
        w_disp = self.image_label.width()
        scale = min(w_disp / w_img, h_disp / h_img)
        off_x = (w_disp - w_img * scale) / 2
        off_y = (h_disp - h_img * scale) / 2

        pts = []
        for p in poly:
            x = int((p.x() - off_x) / scale)
            y = int((p.y() - off_y) / scale)
            x = np.clip(x, 0, w_img - 1)
            y = np.clip(y, 0, h_img - 1)
            pts.append([x, y])

        mask = np.zeros((h_img, w_img), np.uint8)
        cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 1)

        # zero‑out each frame
        if self.raw_image_data.ndim == 2:
            self.raw_image_data[mask == 1] = 0
        else:
            for i in range(self.num_frames):
                raw_frame = (
                    self.raw_image_data[i]
                    if self.raw_image_data.ndim in (3, 4)
                    else self.raw_image_data
                )
                # expand mask for color dims if needed
                if raw_frame.ndim == 3 and raw_frame.shape[2] in (3, 4):
                    m = mask[:, :, None]
                    self.raw_image_data[i] = np.where(m == 1, 0, raw_frame)
                else:
                    self.raw_image_data[i] = np.where(mask == 1, 0, raw_frame)

        self.corrected_image_data = (
            self.raw_image_data.copy()
            if self.raw_image_data.ndim == 2
            else np.stack([f.copy() for f in self.raw_image_data], axis=0)
        )

        self.add_log("Deleted pixels via freeform ROI")
        self.adjust_image()

    # ------------------------------------------------------------
    # Threshold ROI dialog launcher
    # ------------------------------------------------------------
    def on_threshold_roi(self) -> None:
        if self.current_view_mode != "2D":
            QMessageBox.warning(self, "2D required", "Switch to 2D view first.")
            return
        # pick whatever is shown (adjusted → corrected → raw)

        if self.adjusted_image_data is not None:
            base = self.adjusted_image_data
        elif not self.show_raw and self.corrected_image_data is not None:
            base = self.corrected_image_data
        else:
            base = self.raw_image_data
        # check if we have an image
        if base is None:
            QMessageBox.warning(self, "No image", "Load an image first.")
            return
        # prepare gray stack
        if base.ndim == 2:
            volume = base
        elif base.ndim == 3 and base.shape[2] in (3, 4):
            volume = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)
        elif base.ndim == 3:
            volume = base
        elif base.ndim == 4:
            volume = np.array([
                cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) if f.ndim == 3 else f
                for f in base])
        else:
            volume = base
        
        # clamp frame index so it's never out of bounds
        depth = volume.shape[0] if volume.ndim>2 else 1
        init_frame = min(self.current_frame_index, depth - 1)

        dlg = ThresholdROIDialog(
            volume,
            initial_frame=init_frame,
            initial_roi_results=self.roi_results,
            colormap=self.colormap,
            parent=self
        )

        if dlg.exec_() == QDialog.Accepted:
            self.roi_results = dlg.roi_results
            self.add_log("Threshold ROI updated")
            self.update_image_display()

    # ------------------------------------------------------------
    # ML classification methods
    # ------------------------------------------------------------
    def ml_pixel_classifier(self) -> None:
        # spinner in
        self.spinner_label.show()
        self.spinner_movie.start()
        QCoreApplication.processEvents()

        # pick whatever is shown (adjusted → corrected → raw)
        if self.adjusted_image_data is not None:
            display_stack = self.adjusted_image_data
        elif not self.show_raw and self.corrected_image_data is not None:
            display_stack = self.corrected_image_data
        else:
            display_stack = self.raw_image_data

        if display_stack is None:
            QMessageBox.warning(self, "No image", "Load an image first.")
            # spinner out on early exit
            self.spinner_movie.stop()
            self.spinner_label.hide()
            return

        frame = self.get_current_frame(display_stack)
        gray  = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame.copy()
        dlg   = MLPixelClassifierDialog(gray, colormap=self.colormap, parent=self)

        if dlg.exec_() == QDialog.Accepted:
            if self.ml_X_train is None:
                self.ml_X_train = dlg.training_data
                self.ml_y_train = dlg.training_labels
            else:
                self.ml_X_train = np.vstack((self.ml_X_train, dlg.training_data))
                self.ml_y_train = np.hstack((self.ml_y_train, dlg.training_labels))
            self.ml_training_data[self.current_frame_index] = (
                dlg.training_data, dlg.training_labels
            )

            clf = RandomForestClassifier(n_estimators=50)
            clf.fit(self.ml_X_train, self.ml_y_train)
            self.ml_classifier = clf

            mask = (dlg.result_segmentation[:, :, 0] == 0).astype(np.uint8) * 255
            self.roi_results[self.current_frame_index] = {"mask": mask, "contour": None}

            self.update_image_display()
            QMessageBox.information(self, "Segmentation", "ML segmentation applied.")
            self.update_ml_slice_menu()

        # spinner out
        self.spinner_movie.stop()
        self.spinner_label.hide()

    def ml_classify_slice(self, frame_index: int) -> None:
        if frame_index not in self.ml_training_data:
            QMessageBox.warning(self, "No data", f"No training for frame {frame_index}")
            return
        X, y = self.ml_training_data[frame_index]
        clf = RandomForestClassifier(n_estimators=50); clf.fit(X, y)
        base = self.corrected_image_data if not self.show_raw else self.raw_image_data
        if base is None:
            return
        n = 1 if base.ndim in (2, 3) and (base.ndim == 2 or base.shape[2] in (3, 4)) else base.shape[0]
        for i in range(n):
            slice_i = base if n == 1 else base[i]
            gray    = cv2.cvtColor(slice_i, cv2.COLOR_RGB2GRAY) if slice_i.ndim == 3 else slice_i.copy()
            h, w    = gray.shape
            pred    = clf.predict(gray.reshape(-1, 1).astype(np.float32)).reshape(h, w)
            self.roi_results[i] = {"mask": (pred == 1).astype(np.uint8) * 255, "contour": None}
        self.update_image_display()
        QMessageBox.information(self, "Slice Classification", f"Applied frame {frame_index}")

    def ml_classify_stack_combined(self) -> None:
        if self.ml_classifier is None:
            QMessageBox.warning(self, "No classifier", "Run ML Pixel Classifier first.")
            return

        # spinner in
        self.spinner_label.show()
        self.spinner_movie.start()
        QCoreApplication.processEvents()

        base = self.corrected_image_data if not self.show_raw else self.raw_image_data
        if base is None:
            QMessageBox.warning(self, "No image", "Load an image first.")
            # spinner out
            self.spinner_movie.stop()
            self.spinner_label.hide()
            return

        n = 1 if base.ndim in (2, 3) and (base.ndim == 2 or base.shape[2] in (3, 4)) else base.shape[0]
        for i in range(n):
            slice_i = base if n == 1 else base[i]
            gray    = cv2.cvtColor(slice_i, cv2.COLOR_RGB2GRAY) if slice_i.ndim == 3 else slice_i.copy()
            h, w    = gray.shape
            pred    = self.ml_classifier.predict(gray.reshape(-1, 1).astype(np.float32)).reshape(h, w)
            self.roi_results[i] = {"mask": (pred == 1).astype(np.uint8) * 255, "contour": None}

        self.update_image_display()
        QMessageBox.information(self, "Stack Classification", "Applied combined training")

        # spinner out
        self.spinner_movie.stop()
        self.spinner_label.hide()

    def update_ml_slice_menu(self) -> None:
        self.ml_slice_menu.clear()
        for f in sorted(self.ml_training_data):
            act = QAction(f"Frame {f}", self)
            act.triggered.connect(lambda _, fr=f: self.ml_classify_slice(fr))
            self.ml_slice_menu.addAction(act)

    # ------------------------------------------------------------
    # Surface selection tools
    # ------------------------------------------------------------
    def filter_surfaces(self) -> None:
        if self.current_polydata is None:
            QMessageBox.warning(self, "No surface", "Load or generate a surface first.")
            return
        dlg = FilterSurfacesDialog(self.current_polydata, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            self.current_polydata = dlg.filtered_polydata
            mapper = vtk.vtkPolyDataMapper(); mapper.SetInputData(self.current_polydata); mapper.ScalarVisibilityOff()
            actor  = vtk.vtkActor(); actor.SetMapper(mapper); actor.GetProperty().SetColor(1, 0, 0)
            self.vtk_renderer.RemoveAllViewProps(); self.vtk_renderer.AddActor(actor)
            self.vtk_renderer.ResetCamera(); self.vtk_widget.GetRenderWindow().Render()

    def delete_surfaces(self) -> None:
        if self.current_polydata is None:
            QMessageBox.warning(self, "No surface", "Load or generate a surface first.")
            return
        dlg = FilterSurfacesDialog(self.current_polydata, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            self.current_polydata = dlg.filtered_polydata
            mapper = vtk.vtkPolyDataMapper(); mapper.SetInputData(self.current_polydata); mapper.ScalarVisibilityOff()
            actor  = vtk.vtkActor(); actor.SetMapper(mapper); actor.GetProperty().SetColor(1, 0.5, 0.5)
            self.vtk_renderer.RemoveAllViewProps(); self.vtk_renderer.AddActor(actor)
            self.vtk_renderer.ResetCamera(); self.vtk_widget.GetRenderWindow().Render()
            self.add_log("Deleted surfaces outside selected range")

    def clean_surfaces(self) -> None:
        if self.current_polydata is None:
            QMessageBox.warning(self, "No surface", "Load or generate a surface first.")
            return

        # 1) rebuild your 3D boolean volume from self.roi_results
        vol_bool = []
        mask0 = next(iter(self.roi_results.values()))["mask"]
        for z in range(self.num_frames):
            vol_bool.append(self.roi_results.get(z, {}).get("mask", mask0*0) > 0)
        vol_bool = np.stack(vol_bool, axis=0)

        # 2) build a 3×3×3 connectivity — you can upsize this
        #    based on your PSF_x/y/z (in voxels) if you like
        struct = generate_binary_structure(3, 1)

        # 3) close every gap smaller than the structuring element
        closed = binary_closing(vol_bool, structure=struct)

        # 4) re‐threshold to 8‑bit for marching cubes
        vol8 = (closed * 255).astype(np.uint8)

        # 5) import & marching cubes exactly as in render_roi_surface
        importer = vtk.vtkImageImport()
        data = vol8.tobytes()
        importer.CopyImportVoidPointer(data, len(data))
        importer.SetDataScalarTypeToUnsignedChar()
        importer.SetNumberOfScalarComponents(1)
        d0, d1, d2 = vol8.shape[::-1]
        importer.SetDataExtent(0, d0-1, 0, d1-1, 0, d2-1)
        importer.SetWholeExtent(0, d0-1, 0, d1-1, 0, d2-1)
        importer.Update()

        mc = vtk.vtkMarchingCubes()
        mc.SetInputConnection(importer.GetOutputPort())
        mc.SetValue(0, 127)
        mc.Update()

        merged_poly = mc.GetOutput()

        # 6) swap in & re‑render
        self.current_polydata = merged_poly
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(merged_poly)
        mapper.ScalarVisibilityOff()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1, 0.5, 0.5)

        self.vtk_renderer.RemoveAllViewProps()
        self.vtk_renderer.AddActor(actor)
        self.vtk_renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

        self.add_log("Cleaned surfaces by 3D closing")

    # ------------------------------------------------------------
    # 3D rendering
    # ------------------------------------------------------------
    def render_roi_surface(self) -> None:
        # spinner in
        self.spinner_label.show()
        self.spinner_movie.start()
        QCoreApplication.processEvents()

        if not self.roi_results:
            QMessageBox.warning(self, "No ROI", "Create ROIs first.")
            # spinner out
            self.spinner_movie.stop()
            self.spinner_label.hide()
            return

        # build 3D mask volume
        vol = []
        mask0 = next(iter(self.roi_results.values()))["mask"]
        for i in range(self.num_frames):
            mask = self.roi_results.get(i, {}).get("mask", mask0 * 0)
            vol.append(mask)
        vol = np.stack(vol, axis=0).astype(np.uint8)

        importer = vtk.vtkImageImport()
        data = vol.tobytes()
        importer.CopyImportVoidPointer(data, len(data))
        importer.SetDataScalarTypeToUnsignedChar()
        importer.SetNumberOfScalarComponents(1)
        d0, d1, d2 = vol.shape[::-1]
        importer.SetDataExtent(0, d0-1, 0, d1-1, 0, d2-1)
        importer.SetWholeExtent(0, d0-1, 0, d1-1, 0, d2-1)
        importer.Update()

        mc = vtk.vtkMarchingCubes()
        mc.SetInputConnection(importer.GetOutputPort())
        mc.SetValue(0, 127)
        mc.Update()

        poly = mc.GetOutput()
        mapper = vtk.vtkPolyDataMapper(); mapper.SetInputData(poly); mapper.ScalarVisibilityOff()
        actor  = vtk.vtkActor(); actor.SetMapper(mapper); actor.GetProperty().SetColor(1, 0.5, 0.5)

        self.vtk_renderer.RemoveAllViewProps()
        self.vtk_renderer.AddActor(actor)
        self.vtk_renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

        self.current_polydata = poly

        # spinner out
        self.spinner_movie.stop()
        self.spinner_label.hide()

    def render_final_structure(self) -> None:
        # spinner in
        self.spinner_label.show()
        self.spinner_movie.start()
        QCoreApplication.processEvents()

        if not self.roi_results:
            QMessageBox.warning(self, "No ROI", "Create ROIs first.")
            # spinner out
            self.spinner_movie.stop()
            self.spinner_label.hide()
            return

        vol_bool = []
        mask0 = next(iter(self.roi_results.values()))["mask"]
        for i in range(self.num_frames):
            mask = self.roi_results.get(i, {}).get("mask", mask0 * 0)
            vol_bool.append(mask > 0)
        vol_bool = np.stack(vol_bool, axis=0)
        closed = binary_closing(vol_bool, np.ones((3,3,3)))
        filled = binary_fill_holes(closed)
        vol8 = (filled * 255).astype(np.uint8)

        importer = vtk.vtkImageImport()
        data = vol8.tobytes()
        importer.CopyImportVoidPointer(data, len(data))
        importer.SetDataScalarTypeToUnsignedChar()
        importer.SetNumberOfScalarComponents(1)
        d0, d1, d2 = vol8.shape[::-1]
        importer.SetDataExtent(0, d0-1, 0, d1-1, 0, d2-1)
        importer.SetWholeExtent(0, d0-1, 0, d1-1, 0, d2-1)
        importer.Update()

        mc = vtk.vtkMarchingCubes()
        mc.SetInputConnection(importer.GetOutputPort())
        mc.SetValue(0, 127)
        mc.Update()

        poly = mc.GetOutput()
        mapper = vtk.vtkPolyDataMapper(); mapper.SetInputData(poly); mapper.ScalarVisibilityOff()
        actor  = vtk.vtkActor(); actor.SetMapper(mapper); actor.GetProperty().SetColor(1, 1, 0)

        self.vtk_renderer.RemoveAllViewProps()
        self.vtk_renderer.AddActor(actor)
        self.vtk_renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

        self.final_polydata = poly

        # spinner out
        self.spinner_movie.stop()
        self.spinner_label.hide()

    # ------------------------------------------------------------
    # Image adjustment / display
    # ------------------------------------------------------------
    def adjust_image(self) -> None:
        if self.raw_image_data is None:
            return
        base = self.raw_image_data if self.show_raw else self.corrected_image_data
        frame = base if base.ndim == 2 else base[self.current_frame_index]
        f = frame.astype(np.float32)
        self.brightness = self.brightness_slider.value()
        self.contrast   = self.contrast_slider.value() / 100.0
        adj = np.clip(self.contrast * f + self.brightness, 0, 255).astype(np.uint8)
        self.adjusted_image_data = adj
        self.update_image_display()
        self.add_log(f"Brightness={self.brightness}, Contrast={self.contrast}")
        if self.current_view_mode == "3D" and self.num_frames > 1:
            self.update_3d_view()

    def update_image_display(self) -> None:
        if self.adjusted_image_data is None:
            return

        # 1) grab the exact slice the user is looking at
        if self.adjusted_image_data.ndim == 2:
            frame = self.adjusted_image_data
        else:
            frame = self.adjusted_image_data[self.current_frame_index]

        # 2) if it’s single‑channel, apply your chosen OpenCV LUT; otherwise just copy
        if frame.ndim == 2:
            # returns a 3‑channel BGR image
            rgb = cv2.applyColorMap(frame, self.colormap)
        else:
            rgb = frame.copy()

        # 3) overlay any ROI contour
        if self.current_frame_index in self.roi_results:
            cnt = self.roi_results[self.current_frame_index]["contour"]
            if cnt is not None:
                cv2.drawContours(rgb, [cnt], -1, (0,255,255), 2)

        # 4) convert BGR→RGB for Qt
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # 5) push into a QPixmap and scale
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, 3*w, QImage.Format_RGB888)
        pix  = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(
            pix.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        )

    def update_3d_view(self) -> None:
        # spinner in
        self.spinner_label.show()
        self.spinner_movie.start()
        QCoreApplication.processEvents()

        self.set_status("Rendering 3D...", 2000)
        base = self.raw_image_data if self.show_raw else self.corrected_image_data
        if base is None or self.num_frames <= 1:
            # spinner out
            self.spinner_movie.stop()
            self.spinner_label.hide()
            return
        vol = np.clip(self.contrast * base.astype(np.float32) + self.brightness, 0, 255).astype(np.uint8)
        if vol.ndim != 3:
            # spinner out
            self.spinner_movie.stop()
            self.spinner_label.hide()
            return
        

        # upload to VTK
        importer = vtk.vtkImageImport()
        data = vol.tobytes()
        importer.CopyImportVoidPointer(data, len(data))
        importer.SetDataScalarTypeToUnsignedChar()
        importer.SetNumberOfScalarComponents(1)
        d0, d1, d2 = vol.shape[::-1]
        importer.SetDataExtent(0, d0-1, 0, d1-1, 0, d2-1)
        importer.SetWholeExtent(0, d0-1, 0, d1-1, 0, d2-1)
        importer.Update()

        perm = vtk.vtkImagePermute()
        perm.SetInputConnection(importer.GetOutputPort())
        perm.SetFilteredAxes(2, 1, 0)
        perm.Update()

        flip = vtk.vtkImageFlip()
        flip.SetFilteredAxis(1)
        flip.SetInputConnection(perm.GetOutputPort())
        flip.Update()

        conn = flip.GetOutputPort()
        if self.use_gpu and self.is_gpu_supported():
            mapper = vtk.vtkGPUVolumeRayCastMapper()
        else:
            mapper = vtk.vtkSmartVolumeMapper()
        mapper.SetInputConnection(conn)

        prop = vtk.vtkVolumeProperty()
        prop.ShadeOn()
        prop.SetAmbient(0.3)
        prop.SetDiffuse(0.7)
        prop.SetSpecular(0.3)
        prop.SetInterpolationTypeToLinear()

        # set up opacity transfer function
        otf = vtk.vtkPiecewiseFunction()
        for x, y in [(0, 0), (32, 0.05), (96, 0.15), (160, 0.4), (255, 0.8)]:
            otf.AddPoint(x, y)
        prop.SetScalarOpacity(otf)

        # ——— new LUT‑based color transfer function ———
        # build a full 0–255 ColorTransferFunction from your OpenCV colormap
        lut_vals = cv2.applyColorMap(
            np.arange(256, dtype=np.uint8)[:, None],
            self.colormap
        )[:, 0]  # shape (256,3) in BGR
        ctf = vtk.vtkColorTransferFunction()
        for i in range(256):
            b, g, r = lut_vals[i]
            ctf.AddRGBPoint(i, r/255.0, g/255.0, b/255.0)
        prop.SetColor(ctf)
        # ————————————————————————————————

        # create and add the volume
        volume = vtk.vtkVolume()
        volume.SetMapper(mapper)
        volume.SetProperty(prop)

        self.vtk_renderer.RemoveAllViewProps()
        self.vtk_renderer.AddVolume(volume)
        self.vtk_renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()
        self.set_status("3D Ready", 2000)

        # spinner out
        self.spinner_movie.stop()
        self.spinner_label.hide()

    def change_frame(self, val: int) -> None:
        self.current_frame_index = val
        self.z_frame_label.setText(f"Frame: {val+1}/{self.num_frames}")
        if self.current_view_mode == "2D":
            self.adjust_image()

    def get_current_frame(self, data: np.ndarray) -> np.ndarray:
        if data is None:
            return None
        if data.ndim == 2:
            return data
        if data.ndim == 3 and data.shape[2] in (3, 4):
            return data
        if data.ndim == 3:
            return data[self.current_frame_index]
        if data.ndim == 4:
            return data[self.current_frame_index]
        return data

    def handle_mouse_move(self, a0) -> None:
        if self.adjusted_image_data is None:
            self.pixel_value_label.setText("Pixel: N/A")
            return
        pos = a0.pos()
        pix = self.image_label.pixmap()
        if pix is None:
            return
        h_img, w_img = self.adjusted_image_data.shape[:2]
        h_disp, w_disp = self.image_label.height(), self.image_label.width()
        scale = min(w_disp / w_img, h_disp / h_img)
        off_x = (w_disp - w_img * scale) / 2
        off_y = (h_disp - h_img * scale) / 2
        x = int((pos.x() - off_x) / scale)
        y = int((pos.y() - off_y) / scale)
        if not (0 <= x < w_img and 0 <= y < h_img):
            self.pixel_value_label.setText("Pixel: N/A")
            return
        val = (self.adjusted_image_data[y, x]
               if self.adjusted_image_data.ndim == 2
               else self.adjusted_image_data[y, x])
        self.pixel_value_label.setText(f"Pixel ({x},{y}): {val}")

    def reset_adjustments(self) -> None:
        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(100)
        self.add_log("Reset brightness/contrast")
        self.adjust_image()

    def auto_adjust_image(self) -> None:
        base = self.raw_image_data if self.show_raw else self.corrected_image_data
        if base is None:
            return
        frame = self.get_current_frame(base).astype(np.float32)
        mn, mx = frame.min(), frame.max()
        if mx - mn == 0:
            c, b = 1.0, 0
        else:
            c = 255.0 / (mx - mn)
            b = -mn * c
        new_b = int(np.clip(b, -100, 100))
        new_c = int(np.clip(c * 100, 10, 300))
        self.brightness_slider.setValue(new_b)
        self.contrast_slider.setValue(new_c)
        self.add_log(f"Auto adj B={new_b},C={new_c/100:.2f}")
        self.adjust_image()

    def toggle_raw_corrected(self) -> None:
        self.show_raw = not self.show_raw
        text = "Show Corrected" if self.show_raw else "Show Raw"
        self.toggle_raw_button.setText(text)
        self.add_log(f"Toggled view: {text}")
        self.adjust_image()

    def toggle_view(self) -> None:
        if self.current_view_mode == "2D":
            if self.num_frames > 1 and self.raw_image_data is not None:
                self.current_view_mode = "3D"
                self.display_stack.setCurrentIndex(1)
                self.toggle_button.setText("Switch to 2D View")
                self.update_3d_view()
            else:
                QMessageBox.information(self, "3D View", "Need a z‑stack to view 3D.")
        else:
            self.current_view_mode = "2D"
            self.display_stack.setCurrentIndex(0)
            self.toggle_button.setText("Switch to 3D View")

    def toggle_gpu_acceleration(self, checked: bool) -> None:
        if checked and self.is_gpu_supported():
            self.use_gpu = True
            QMessageBox.information(self, "GPU", "GPU Acceleration enabled.")
        elif checked:
            self.use_gpu = False
            QMessageBox.warning(self, "GPU", "GPU not supported.")
        else:
            self.use_gpu = False
            QMessageBox.information(self, "GPU", "GPU Acceleration disabled.")
        self.add_log(f"GPU set to {self.use_gpu}")

    def close_image(self) -> None:
        self.raw_image_data = None
        self.corrected_image_data = None
        self.adjusted_image_data = None
        self.roi_results.clear()
        self.ml_training_data.clear()
        self.ml_X_train = None
        self.ml_y_train = None
        self.current_polydata = None
        self.final_polydata = None
        self.image_label.clear()
        self.vtk_renderer.RemoveAllViewProps()
        self.vtk_widget.GetRenderWindow().Render()
        self.z_slider.hide()
        self.z_frame_label.setText("Frame: 0/0")
        self.add_log("Image closed")

    def add_log(self, msg: str) -> None:
        self.process_log.append(msg)
        print("LOG:", msg)

# ------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageApp()
    window.show()
    sys.exit(app.exec_())

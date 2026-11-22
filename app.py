import sys
import os
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QComboBox, QPushButton, QSizePolicy, QGridLayout, QStatusBar,
    QLineEdit, QFileDialog, QTabWidget, QScrollArea, QCheckBox, QSlider, QListWidget, QListWidgetItem
)
from PySide6.QtGui import QImage, QPixmap, QFont, QIcon
from PySide6.QtCore import Qt, QThread, Signal, QSize, QTimer

import pyqtgraph as pg
from sklearn.decomposition import PCA
from scipy.spatial import KDTree, ConvexHull
from scipy.stats import gaussian_kde
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.cm as cm
import umap
import imageio

# --- Data Generation Helpers ---
def create_shape_image(shape_type, color, size=64):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cx, cy = size // 2, size // 2
    r = size // 4
    
    if shape_type == 0: # Circle
        yy, xx = np.ogrid[:size, :size]
        mask = (xx - cx)**2 + (yy - cy)**2 <= r**2
    elif shape_type == 1: # Square
        mask = np.zeros((size, size), dtype=bool)
        mask[cx-r:cx+r, cy-r:cy+r] = True
    elif shape_type == 2: # Triangle
        mask = np.zeros((size, size), dtype=bool)
        for i in range(size):
            w = int((i / size) * (r*2))
            if w > 0:
                mask[size - i - 10, cx - w//2 : cx + w//2] = True
    elif shape_type == 3: # Cross
        mask = np.zeros((size, size), dtype=bool)
        mask[cx-r//2:cx+r//2, cy-r:cy+r] = True
        mask[cx-r:cx+r, cy-r//2:cy+r//2] = True
    else: # Diamond
        yy, xx = np.ogrid[:size, :size]
        mask = np.abs(xx - cx) + np.abs(yy - cy) <= r

    img[mask] = color
    noise = np.random.randint(-10, 10, (size, size, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img

class ReductionWorker(QThread):
    finished = Signal(object, object)
    error = Signal(str)

    def __init__(self, data, method):
        super().__init__()
        self.data = data
        self.method = method

    def run(self):
        try:
            reducer = None
            if self.method == "PCA":
                reducer = PCA(n_components=2)
                coords = reducer.fit_transform(self.data)
            elif self.method == "UMAP":
                reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1)
                coords = reducer.fit_transform(self.data)
            else:
                coords = self.data[:, :2]
            self.finished.emit(coords, reducer)
        except Exception as e:
            self.error.emit(str(e))

class MetricsWorker(QThread):
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, data, labels):
        super().__init__()
        self.data = data
        self.labels = labels

    def run(self):
        try:
            metrics = {}
            # Subsample if too large for speed
            n = len(self.data)
            if n > 1000:
                idxs = np.random.choice(n, 1000, replace=False)
                data = self.data[idxs]
                labels = self.labels[idxs]
            else:
                data = self.data
                labels = self.labels
                
            metrics['Silhouette'] = silhouette_score(data, labels)
            metrics['Davies-Bouldin'] = davies_bouldin_score(data, labels)
            metrics['Calinski-Harabasz'] = calinski_harabasz_score(data, labels)
            self.finished.emit(metrics)
        except Exception as e:
            self.error.emit(str(e))

class LatentSpaceViewerProV7(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pro Latent Space Visualizer v7.0 (Analytics & Concept Math)")
        self.resize(1400, 900)
        self.setup_dark_mode()

        # --- 1. Generate Mock Data ---
        self.num_samples = 500
        self.latent_dim = 8
        self.latent_vectors = np.random.randn(self.num_samples, self.latent_dim) * 0.5
        
        self.latent_vectors[:, 0] = np.abs(self.latent_vectors[:, 0]) * 100
        self.latent_vectors[:, 1] = np.abs(self.latent_vectors[:, 1]) * 100
        self.latent_vectors[:, 2] = np.abs(self.latent_vectors[:, 2]) * 100
        self.latent_vectors[:, 3] = np.abs(self.latent_vectors[:, 3]) * 1.5
        
        self.class_names = ["Circle", "Square", "Triangle", "Cross", "Diamond"]
        self.class_colors = [(255,80,80), (80,255,80), (80,80,255), (255,255,80), (255,80,255)]
        self.labels = np.clip(self.latent_vectors[:, 3].astype(int), 0, 4)
        
        self.images = []
        for i in range(self.num_samples):
            c = (int(self.latent_vectors[i, 0]), int(self.latent_vectors[i, 1]), int(self.latent_vectors[i, 2]))
            c = tuple(np.clip(c, 0, 255))
            self.images.append(create_shape_image(self.labels[i], c))

        # State variables
        self.coords_2d = None
        self.kdtree = None
        self.kdtree_highd = None
        self.reducer = None
        self.selections = []
        self.scatter_items = []
        self.colormap = self.get_viridis_colormap()
        self.active_vector = None
        self.heatmap_item = None
        self.roi_item = None
        self.traj_line = None
        self.traj_scatter = None
        self.grid_images = []
        self.centroid_markers = None
        self.class_centroids_highd = []
        
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.step_animation)
        self.anim_t = 0.0
        self.anim_direction = 1
        self.anim_frames = []

        # --- 2. Setup UI ---
        self.setup_ui()
        self.update_plot()
        self.calculate_metrics()

    def get_viridis_colormap(self):
        pos = np.linspace(0, 1, 256)
        colors = cm.get_cmap('viridis')(pos)
        colors = (colors * 255).astype(np.uint8)
        return pg.ColorMap(pos, colors)

    def mock_decode(self, vec):
        r = int(np.clip(vec[0], 0, 255))
        g = int(np.clip(vec[1], 0, 255))
        b = int(np.clip(vec[2], 0, 255))
        shape_idx = int(np.clip(vec[3], 0, 4))
        return create_shape_image(shape_idx, (r, g, b))

    def setup_dark_mode(self):
        pg.setConfigOption('background', '#1e1e1e')
        pg.setConfigOption('foreground', 'w')
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QWidget { color: #d4d4d4; font-size: 12px; }
            QPushButton { background-color: #3a3d41; border: 1px solid #5a5d61; padding: 6px; border-radius: 4px; }
            QPushButton:hover { background-color: #4a4d51; }
            QComboBox { background-color: #3a3d41; border: 1px solid #5a5d61; padding: 4px; border-radius: 4px; }
            QLineEdit { background-color: #3a3d41; border: 1px solid #5a5d61; padding: 4px; border-radius: 4px; }
            QLabel { background-color: transparent; }
            QStatusBar { background-color: #007acc; color: white; }
            QTabWidget::pane { border: 1px solid #5a5d61; }
            QTabBar::tab { background-color: #2d2d2d; padding: 8px 12px; border: 1px solid #5a5d61; border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px; }
            QTabBar::tab:selected { background-color: #3a3d41; color: white; }
            QSlider::groove:horizontal { border: 1px solid #5a5d61; height: 4px; background: #2d2d2d; }
            QSlider::handle:horizontal { background: #007acc; width: 12px; margin: -6px 0; border-radius: 6px; }
            QListWidget { background-color: #1e1e1e; border: 1px solid #5a5d61; }
            QListWidget::item { background-color: #2d2d2d; }
            QListWidget::item:selected { background-color: #007acc; }
        """)

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # --- Left: Plot Area ---
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Component 2')
        self.plot_widget.setLabel('bottom', 'Component 1')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.plot_widget, stretch=3)

        # --- Right: Controls & Tabs ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_panel.setMaximumWidth(520)
        layout.addWidget(right_panel)

        # Top Controls
        controls_layout = QGridLayout()
        controls_layout.addWidget(QLabel("Reduction:"), 0, 0)
        self.method_combo = QComboBox()
        self.method_combo.addItems(["PCA", "UMAP"])
        self.method_combo.currentIndexChanged.connect(self.update_plot)
        controls_layout.addWidget(self.method_combo, 0, 1)

        controls_layout.addWidget(QLabel("Color By:"), 1, 0)
        self.colorby_combo = QComboBox()
        self.colorby_combo.addItems(["Class", "Outlier Score"] + [f"Dim {i}" for i in range(self.latent_dim)])
        self.colorby_combo.currentIndexChanged.connect(self.redraw_scatter)
        controls_layout.addWidget(self.colorby_combo, 1, 1)

        controls_layout.addWidget(QLabel("Isolate Class:"), 2, 0)
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All"] + self.class_names)
        self.filter_combo.currentIndexChanged.connect(self.apply_class_filter)
        controls_layout.addWidget(self.filter_combo, 2, 1)
        
        self.chk_heatmap = QCheckBox("Density Heatmap")
        self.chk_heatmap.stateChanged.connect(self.toggle_heatmap)
        controls_layout.addWidget(self.chk_heatmap, 3, 0)

        self.chk_grid = QCheckBox("Decoder Grid (PCA Only)")
        self.chk_grid.stateChanged.connect(self.toggle_decoder_grid)
        controls_layout.addWidget(self.chk_grid, 3, 1)

        right_layout.addLayout(controls_layout)

        # Action Buttons
        btn_layout = QHBoxLayout()
        self.btn_reset = QPushButton("Reset Sel.")
        self.btn_reset.clicked.connect(self.reset_selection)
        btn_layout.addWidget(self.btn_reset)
        
        self.btn_sample_z = QPushButton("Sample Random Z")
        self.btn_sample_z.clicked.connect(self.sample_random_z)
        btn_layout.addWidget(self.btn_sample_z)
        
        self.btn_roi = QPushButton("ROI Select")
        self.btn_roi.setCheckable(True)
        self.btn_roi.toggled.connect(self.toggle_roi)
        btn_layout.addWidget(self.btn_roi)
        right_layout.addLayout(btn_layout)

        # Search Bar
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Search ID:"))
        self.search_input = QLineEdit()
        self.search_input.returnPressed.connect(self.search_point)
        search_layout.addWidget(self.search_input)
        right_layout.addLayout(search_layout)

        # Tabs
        self.tabs = QTabWidget()
        right_layout.addWidget(self.tabs)

        # Tab 1: Inspector
        tab1 = QWidget()
        tab1_layout = QVBoxLayout(tab1)
        self.tabs.addTab(tab1, "Inspector")
        
        tab1_layout.addWidget(QLabel("<b>Hover Info:</b>"))
        self.info_label = QLabel("Move mouse over points...")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("background-color: #2d2d2d; padding: 10px; border-radius: 5px;")
        tab1_layout.addWidget(self.info_label)

        tab1_layout.addWidget(QLabel("<b>Selected / Sampled:</b>"))
        self.image_label = QLabel("None")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(180, 180)
        self.image_label.setStyleSheet("background-color: #2d2d2d; border: 1px solid #5a5d61; border-radius: 5px;")
        tab1_layout.addWidget(self.image_label)
        tab1_layout.addStretch()

        # Tab 2: Latent Editor
        tab2 = QWidget()
        tab2_layout = QVBoxLayout(tab2)
        self.tabs.addTab(tab2, "Latent Editor")
        
        tab2_layout.addWidget(QLabel("<b>Live Latent Vector:</b> Drag sliders to morph."))
        self.slider_layout = QGridLayout()
        tab2_layout.addLayout(self.slider_layout)
        
        self.btn_save_edit = QPushButton("Save Edited Point to Dataset")
        self.btn_save_edit.clicked.connect(self.save_edited_point)
        tab2_layout.addWidget(self.btn_save_edit)
        tab2_layout.addStretch()
        self.init_sliders()

        # Tab 3: Operations & Animation
        tab3 = QWidget()
        tab3_layout = QVBoxLayout(tab3)
        self.tabs.addTab(tab3, "Operations")
        
        op_layout = QHBoxLayout()
        op_layout.addWidget(QLabel("Operation:"))
        self.op_combo = QComboBox()
        self.op_combo.addItems(["Lerp (A -> B)", "Slerp (A -> B)", "Arithmetic (A + B - C)"])
        self.op_combo.currentIndexChanged.connect(self.reset_selection)
        op_layout.addWidget(self.op_combo)
        tab3_layout.addLayout(op_layout)

        self.btn_pc_traversal = QPushButton("Principal Direction Traversal")
        self.btn_pc_traversal.clicked.connect(self.generate_pc_traversal)
        tab3_layout.addWidget(self.btn_pc_traversal)

        # Animation Controls
        anim_layout = QHBoxLayout()
        self.btn_play_anim = QPushButton("Play Animation")
        self.btn_play_anim.clicked.connect(self.toggle_animation)
        self.btn_play_anim.setEnabled(False)
        anim_layout.addWidget(self.btn_play_anim)
        
        self.btn_export_gif = QPushButton("Export to GIF")
        self.btn_export_gif.clicked.connect(self.export_gif)
        self.btn_export_gif.setEnabled(False)
        anim_layout.addWidget(self.btn_export_gif)
        tab3_layout.addLayout(anim_layout)

        tab3_layout.addWidget(QLabel("<b>Point Results:</b> <small>(Right-click to save)</small>"))
        self.result_grid_layout = QGridLayout()
        tab3_layout.addLayout(self.result_grid_layout)
        
        # Centroid Math Section
        tab3_layout.addWidget(QLabel("<b>Centroid (Concept) Arithmetic:</b>"))
        centroid_math_layout = QHBoxLayout()
        
        self.cmb_c1 = QComboBox()
        self.cmb_c1.addItems(self.class_names)
        centroid_math_layout.addWidget(self.cmb_c1)
        
        self.cmb_op1 = QComboBox()
        self.cmb_op1.addItems(["+", "-"])
        centroid_math_layout.addWidget(self.cmb_op1)
        
        self.cmb_c2 = QComboBox()
        self.cmb_c2.addItems(self.class_names)
        centroid_math_layout.addWidget(self.cmb_c2)
        
        self.cmb_op2 = QComboBox()
        self.cmb_op2.addItems(["-", "+"])
        centroid_math_layout.addWidget(self.cmb_op2)
        
        self.cmb_c3 = QComboBox()
        self.cmb_c3.addItems(self.class_names)
        centroid_math_layout.addWidget(self.cmb_c3)
        
        self.btn_calc_centroids = QPushButton("Compute")
        self.btn_calc_centroids.clicked.connect(self.compute_centroid_math)
        centroid_math_layout.addWidget(self.btn_calc_centroids)
        tab3_layout.addLayout(centroid_math_layout)
        
        self.centroid_result_grid = QGridLayout()
        tab3_layout.addLayout(self.centroid_result_grid)
        
        tab3_layout.addStretch()

        # Tab 4: Dataset Browser
        tab4 = QWidget()
        tab4_layout = QVBoxLayout(tab4)
        self.tabs.addTab(tab4, "Dataset Browser")
        
        self.list_widget = QListWidget()
        self.list_widget.setViewMode(QListWidget.IconMode)
        self.list_widget.setIconSize(QSize(60, 60))
        self.list_widget.setResizeMode(QListWidget.Adjust)
        self.list_widget.itemClicked.connect(self.on_dataset_item_clicked)
        tab4_layout.addWidget(self.list_widget)
        self.populate_dataset_browser()

        # Tab 5: KNN Inspector
        tab5 = QWidget()
        tab5_layout = QVBoxLayout(tab5)
        self.tabs.addTab(tab5, "KNN Inspector")
        
        tab5_layout.addWidget(QLabel("<b>5 Nearest Neighbors (High-D):</b>"))
        self.knn_grid_layout = QGridLayout()
        tab5_layout.addLayout(self.knn_grid_layout)
        tab5_layout.addStretch()
        
        # Tab 6: Analytics
        tab6 = QWidget()
        tab6_layout = QVBoxLayout(tab6)
        self.tabs.addTab(tab6, "Analytics")
        
        tab6_layout.addWidget(QLabel("<b>Latent Space Clustering Metrics:</b>"))
        self.metrics_label = QLabel("Computing...")
        self.metrics_label.setWordWrap(True)
        self.metrics_label.setStyleSheet("background-color: #2d2d2d; padding: 10px; border-radius: 5px; font-family: monospace;")
        tab6_layout.addWidget(self.metrics_label)
        
        self.btn_recalc_metrics = QPushButton("Recalculate Metrics")
        self.btn_recalc_metrics.clicked.connect(self.calculate_metrics)
        tab6_layout.addWidget(self.btn_recalc_metrics)
        tab6_layout.addStretch()

        self.statusBar().showMessage("Ready. Select Point A.")

        # Visual Markers
        self.marker_a = pg.ScatterPlotItem(size=25, pen=pg.mkPen('w', width=2), brush=pg.mkBrush(None), symbol='o')
        self.marker_b = pg.ScatterPlotItem(size=25, pen=pg.mkPen('c', width=2), brush=pg.mkBrush(None), symbol='o')
        self.marker_c = pg.ScatterPlotItem(size=25, pen=pg.mkPen('m', width=2), brush=pg.mkBrush(None), symbol='o')
        self.marker_search = pg.ScatterPlotItem(size=30, pen=pg.mkPen('y', width=2), brush=pg.mkBrush(None), symbol='star')
        self.marker_random = pg.ScatterPlotItem(size=30, pen=pg.mkPen('r', width=2), brush=pg.mkBrush(None), symbol='x')
        self.path_line = pg.PlotCurveItem(pen=pg.mkPen('w', width=1, style=Qt.DashLine))
        
        self.traj_line = pg.PlotCurveItem(pen=pg.mkPen('c', width=2))
        self.traj_scatter = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush('c'), symbol='o')
        
        self.centroid_markers = pg.ScatterPlotItem(size=40, pen=pg.mkPen('w', width=1.5), brush=pg.mkBrush(None), symbol='star')
        
        self.plot_widget.addItem(self.marker_a)
        self.plot_widget.addItem(self.marker_b)
        self.plot_widget.addItem(self.marker_c)
        self.plot_widget.addItem(self.marker_search)
        self.plot_widget.addItem(self.marker_random)
        self.plot_widget.addItem(self.path_line)
        self.plot_widget.addItem(self.traj_line)
        self.plot_widget.addItem(self.traj_scatter)
        self.plot_widget.addItem(self.centroid_markers)
        self.hide_markers()

    def calculate_metrics(self):
        self.metrics_label.setText("Computing metrics in background...")
        self.worker_metrics = MetricsWorker(self.latent_vectors, self.labels)
        self.worker_metrics.finished.connect(self.on_metrics_finished)
        self.worker_metrics.error.connect(self.on_metrics_error)
        self.worker_metrics.start()

    def on_metrics_error(self, err):
        self.metrics_label.setText(f"Error: {err}")

    def on_metrics_finished(self, metrics):
        text = (
            f"Silhouette Score:      {metrics['Silhouette']:.4f}\n"
            f"  (Range: -1 to 1. Higher is better. Measures cluster density & separation)\n\n"
            f"Davies-Bouldin Index:  {metrics['Davies-Bouldin']:.4f}\n"
            f"  (Lower is better. Measures cluster similarity)\n\n"
            f"Calinski-Harabasz:     {metrics['Calinski-Harabasz']:.4f}\n"
            f"  (Higher is better. Measures variance ratio)"
        )
        self.metrics_label.setText(text)

    def populate_dataset_browser(self):
        for i, img_arr in enumerate(self.images):
            h, w, c = img_arr.shape
            q_img = QImage(img_arr.data, w, h, 3 * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            item = QListWidgetItem(QIcon(pixmap), str(i))
            self.list_widget.addItem(item)

    def on_dataset_item_clicked(self, item):
        idx = int(item.text())
        self.search_input.setText(str(idx))
        self.search_point()

    def init_sliders(self):
        self.sliders = []
        self.slider_labels = []
        for i in range(self.latent_dim):
            lbl = QLabel(f"Dim {i}: 0.00")
            self.slider_labels.append(lbl)
            self.slider_layout.addWidget(lbl, i, 0)
            
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(-200)
            slider.setMaximum(200)
            slider.setValue(0)
            slider.setEnabled(False)
            slider.valueChanged.connect(lambda val, idx=i: self.on_slider_changed(idx, val))
            self.sliders.append(slider)
            self.slider_layout.addWidget(slider, i, 1)

    def update_sliders(self, vec):
        for i, val in enumerate(vec):
            self.sliders[i].blockSignals(True)
            self.sliders[i].setEnabled(True)
            self.sliders[i].setValue(int(val))
            self.slider_labels[i].setText(f"Dim {i}: {val:.2f}")
            self.sliders[i].blockSignals(False)

    def on_slider_changed(self, idx, val):
        if self.active_vector is None: return
        self.active_vector[idx] = float(val)
        self.slider_labels[idx].setText(f"Dim {idx}: {float(val):.2f}")
        img = self.mock_decode(self.active_vector)
        self.display_image(img, self.image_label, size=180)

    def save_edited_point(self):
        if self.active_vector is not None:
            self.latent_vectors = np.vstack([self.latent_vectors, self.active_vector])
            new_img = self.mock_decode(self.active_vector)
            self.images.append(new_img)
            new_label = int(np.clip(self.active_vector[3], 0, 4))
            self.labels = np.append(self.labels, new_label)
            self.num_samples += 1
            
            h, w, c = new_img.shape
            q_img = QImage(new_img.data, w, h, 3 * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            item = QListWidgetItem(QIcon(pixmap), str(self.num_samples - 1))
            self.list_widget.addItem(item)
            
            self.statusBar().showMessage("Edited point saved. Re-run PCA/UMAP to see it plotted.")
            self.update_plot()

    def hide_markers(self):
        self.marker_a.setData([], [])
        self.marker_b.setData([], [])
        self.marker_c.setData([], [])
        self.marker_search.setData([], [])
        self.marker_random.setData([], [])
        self.path_line.setData([], [])
        self.traj_line.setData([], [])
        self.traj_scatter.setData([], [])
        self.centroid_markers.setData([], [])

    def toggle_roi(self, checked):
        if checked:
            if self.coords_2d is None:
                self.btn_roi.setChecked(False)
                return
            x_min, x_max = self.coords_2d[:, 0].min(), self.coords_2d[:, 0].max()
            y_min, y_max = self.coords_2d[:, 1].min(), self.coords_2d[:, 1].max()
            
            self.roi_item = pg.RectROI([x_min, y_min], [(x_max-x_min)/3, (y_max-y_min)/3], pen=pg.mkPen('y', width=2))
            self.roi_item.sigRegionChangeFinished.connect(self.compute_roi_mean)
            self.plot_widget.addItem(self.roi_item)
            self.compute_roi_mean()
        else:
            if self.roi_item:
                self.plot_widget.removeItem(self.roi_item)
                self.roi_item = None

    def compute_roi_mean(self):
        if not self.roi_item or self.coords_2d is None: return
        pos = self.roi_item.pos()
        size = self.roi_item.size()
        x0, y0 = pos.x(), pos.y()
        x1, y1 = x0 + size.x(), y0 + size.y()
        
        mask = (self.coords_2d[:, 0] >= x0) & (self.coords_2d[:, 0] <= x1) & \
               (self.coords_2d[:, 1] >= y0) & (self.coords_2d[:, 1] <= y1)
               
        idxs = np.where(mask)[0]
        if len(idxs) > 0:
            mean_vec = np.mean(self.latent_vectors[idxs], axis=0)
            self.active_vector = mean_vec
            img = self.mock_decode(mean_vec)
            self.display_image(img, self.image_label, size=180)
            self.update_sliders(mean_vec)
            self.tabs.setCurrentIndex(1)
            self.statusBar().showMessage(f"ROI contains {len(idxs)} points. Displaying mean latent vector.")

    def update_plot(self):
        method = self.method_combo.currentText()
        self.statusBar().showMessage(f"Computing {method}...")
        self.method_combo.setEnabled(False)

        self.worker = ReductionWorker(self.latent_vectors, method)
        self.worker.finished.connect(self.on_reduction_finished)
        self.worker.error.connect(self.on_reduction_error)
        self.worker.start()

    def on_reduction_error(self, err_msg):
        self.statusBar().showMessage(f"Error: {err_msg}")
        self.method_combo.setEnabled(True)

    def on_reduction_finished(self, coords, reducer):
        self.coords_2d = coords
        self.kdtree = KDTree(coords)
        self.kdtree_highd = KDTree(self.latent_vectors)
        self.reducer = reducer
        
        # Compute high-d centroids
        self.class_centroids_highd = []
        for i in range(5):
            idxs = np.where(self.labels == i)[0]
            self.class_centroids_highd.append(np.mean(self.latent_vectors[idxs], axis=0))
            
        self.redraw_scatter()
        if self.chk_heatmap.isChecked():
            self.toggle_heatmap(True)
        if self.chk_grid.isChecked():
            self.toggle_decoder_grid(True)

    def toggle_heatmap(self, state):
        if state:
            if self.coords_2d is None: return
            if self.heatmap_item: self.plot_widget.removeItem(self.heatmap_item)
            
            x, y = self.coords_2d[:, 0], self.coords_2d[:, 1]
            kde = gaussian_kde(np.vstack([x, y]))
            xlim = (x.min(), x.max())
            ylim = (y.min(), y.max())
            xx, yy = np.mgrid[xlim[0]:xlim[1]:100j, ylim[0]:ylim[1]:100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            f = np.reshape(kde(positions).T, xx.shape)
            
            img_item = pg.ImageItem()
            img_item.setImage(f.T)
            img_item.setRect(xlim[0], ylim[0], xlim[1]-xlim[0], ylim[1]-ylim[0])
            
            pos = np.linspace(0, 1, 256)
            colors = cm.get_cmap('inferno')(pos)
            colors = (colors * 255).astype(np.uint8)
            cmap = pg.ColorMap(pos, colors)
            img_item.setColorMap(cmap)
            img_item.setOpacity(0.5)
            
            self.heatmap_item = img_item
            self.plot_widget.addItem(self.heatmap_item)
            self.heatmap_item.setZValue(-100)
        else:
            if self.heatmap_item:
                self.plot_widget.removeItem(self.heatmap_item)
                self.heatmap_item = None

    def toggle_decoder_grid(self, state):
        for img_item in self.grid_images:
            self.plot_widget.removeItem(img_item)
        self.grid_images = []
        
        if state:
            if self.coords_2d is None or self.method_combo.currentText() != "PCA":
                self.statusBar().showMessage("Decoder Grid only works with PCA.")
                self.chk_grid.setChecked(False)
                return
                
            x_min, x_max = self.coords_2d[:, 0].min(), self.coords_2d[:, 0].max()
            y_min, y_max = self.coords_2d[:, 1].min(), self.coords_2d[:, 1].max()
            
            xs = np.linspace(x_min + (x_max-x_min)*0.1, x_max - (x_max-x_min)*0.1, 5)
            ys = np.linspace(y_min + (y_max-y_min)*0.1, y_max - (y_max-y_min)*0.1, 5)
            
            img_size = (x_max - x_min) / 12
            
            for x in xs:
                for y in ys:
                    vec = self.reducer.inverse_transform([[x, y]])[0]
                    img_arr = self.mock_decode(vec)
                    
                    h, w, c = img_arr.shape
                    q_img = QImage(img_arr.data, w, h, 3 * w, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_img)
                    
                    img_item = pg.ImageItem(pixmap.toImage())
                    img_item.setRect(x - img_size/2, y - img_size/2, img_size, img_size)
                    img_item.setOpacity(0.8)
                    self.plot_widget.addItem(img_item)
                    self.grid_images.append(img_item)
                    
            self.statusBar().showMessage("Decoder Grid generated.")

    def redraw_scatter(self):
        self.plot_widget.clear()
        
        items_to_re_add = [self.heatmap_item, self.marker_a, self.marker_b, self.marker_c, 
                           self.marker_search, self.marker_random, self.path_line, 
                           self.traj_line, self.traj_scatter, self.centroid_markers]
        for item in items_to_re_add:
            if item: self.plot_widget.addItem(item)
        
        if self.heatmap_item: self.heatmap_item.setZValue(-100)
        self.hide_markers()
        
        symbols = ['o', 's', 't', '+', 'd']
        self.scatter_items = []
        color_mode = self.colorby_combo.currentText()
        
        if color_mode == "Class":
            for i in range(5):
                idxs = np.where(self.labels == i)[0]
                pts = self.coords_2d[idxs]
                if len(pts) >= 3:
                    try:
                        hull = ConvexHull(pts)
                        hull_pts = pts[hull.vertices]
                        hull_pts = np.vstack([hull_pts, hull_pts[0]])
                        poly = pg.PolygonItem(hull_pts, pen=pg.mkPen(self.class_colors[i], width=1.5), brush=pg.mkBrush(self.class_colors[i] + (40,)))
                        self.plot_widget.addItem(poly)
                    except Exception: pass

        if color_mode == "Outlier Score":
            dists = np.zeros(self.num_samples)
            for i in range(self.num_samples):
                c = self.class_centroids_highd[self.labels[i]]
                dists[i] = np.linalg.norm(self.latent_vectors[i] - c)
            norm_dists = (dists - dists.min()) / (dists.max() - dists.min() + 1e-8)

        for i in range(5):
            idxs = np.where(self.labels == i)[0]
            pts = self.coords_2d[idxs]
            
            if color_mode == "Class":
                color = pg.mkColor(self.class_colors[i])
                brushes = [pg.mkBrush(color)] * len(idxs)
            elif color_mode == "Outlier Score":
                colors = cm.get_cmap('coolwarm')(norm_dists[idxs])
                colors = (colors * 255).astype(np.uint8)
                brushes = [pg.mkBrush(c) for c in colors]
            else:
                dim_idx = int(color_mode.split(" ")[1])
                vals = self.latent_vectors[idxs, dim_idx]
                norm_vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
                colors = self.colormap.map(norm_vals)
                brushes = [pg.mkBrush(c) for c in colors]

            scatter = pg.ScatterPlotItem(
                x=pts[:, 0], y=pts[:, 1], symbol=symbols[i], size=8,
                pen=pg.mkPen(None), brush=brushes, data=idxs
            )
            self.plot_widget.addItem(scatter)
            self.scatter_items.append(scatter)
            scatter.sigClicked.connect(self.on_point_clicked)
            scatter.sigMouseHover.connect(self.on_points_hovered)

        cen_x, cen_y = [], []
        for i in range(5):
            idxs = np.where(self.labels == i)[0]
            cen_x.append(np.mean(self.coords_2d[idxs, 0]))
            cen_y.append(np.mean(self.coords_2d[idxs, 1]))
        self.centroid_markers.setData(x=cen_x, y=cen_y)

        self.method_combo.setEnabled(True)
        self.reset_selection()
        self.apply_class_filter()
        
        if self.chk_grid.isChecked():
            self.toggle_decoder_grid(True)

    def apply_class_filter(self):
        target_class = self.filter_combo.currentIndex() - 1
        for i, scatter in enumerate(self.scatter_items):
            if target_class == -1 or i == target_class:
                scatter.setOpacity(1.0)
            else:
                scatter.setOpacity(0.05)

    def on_points_hovered(self, items):
        if not items:
            self.info_label.setText("Move mouse over points...")
            return
        pt = items[0]
        idx = pt.data()
        label = self.labels[idx]
        vec = self.latent_vectors[idx]
        
        # Distance to centroid in high-d space
        cen_vec = self.class_centroids_highd[label]
        dist_to_cen = np.linalg.norm(vec - cen_vec)
        
        self.info_label.setText(
            f"<b>Point ID:</b> {idx}<br>"
            f"<b>Class:</b> {self.class_names[label]}<br>"
            f"<b>2D Coord:</b> ({self.coords_2d[idx, 0]:.2f}, {self.coords_2d[idx, 1]:.2f})<br>"
            f"<b>Dist to Centroid:</b> {dist_to_cen:.2f}<br>"
            f"<b>Vec:</b> [{', '.join([f'{v:.1f}' for v in vec[:4]])}...]"
        )

    def on_point_clicked(self, plot_item, points):
        if not points: return
        pt = points[0]
        idx = pt.data()
        
        self.active_vector = self.latent_vectors[idx].copy()
        self.display_image(self.images[idx], self.image_label, size=180)
        self.update_sliders(self.active_vector)
        self.compute_knn(idx)
        self.tabs.setCurrentIndex(1)
        
        self.selections.append(idx)
        
        x, y = self.coords_2d[idx]
        if len(self.selections) == 1:
            self.marker_a.setData([x], [y])
            self.statusBar().showMessage(f"Point A selected (ID: {idx}). Select Point B.")
        elif len(self.selections) == 2:
            self.marker_b.setData([x], [y])
            self.statusBar().showMessage(f"Point B selected (ID: {idx}). Computing operation...")
            self.update_path_line()
            self.process_operation()
        elif len(self.selections) == 3:
            self.marker_c.setData([x], [y])
            self.statusBar().showMessage(f"Point C selected (ID: {idx}). Computing operation...")
            self.update_path_line()
            self.process_operation()

    def compute_knn(self, idx):
        self.clear_layout(self.knn_grid_layout)
        
        vec = self.latent_vectors[idx]
        dists, idxs = self.kdtree_highd.query(vec, k=6)
        
        for i in range(1, 6):
            n_idx = idxs[i]
            n_img = self.images[n_idx]
            n_dist = dists[i]
            
            h, w, c = n_img.shape
            q_img = QImage(n_img.data, w, h, 3 * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            
            container = QWidget()
            vbox = QVBoxLayout(container)
            vbox.setContentsMargins(0, 0, 0, 0)
            lbl_img = QLabel()
            lbl_img.setPixmap(pixmap.scaled(70, 70, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            lbl_img.setAlignment(Qt.AlignCenter)
            lbl_img.setStyleSheet("background-color: #2d2d2d; border: 1px solid #5a5d61; border-radius: 3px;")
            
            lbl_text = QLabel(f"ID: {n_idx}<br>Dist: {n_dist:.2f}")
            lbl_text.setAlignment(Qt.AlignCenter)
            lbl_text.setStyleSheet("font-size: 10px; color: #aaa;")
            
            vbox.addWidget(lbl_img)
            vbox.addWidget(lbl_text)
            self.knn_grid_layout.addWidget(container, 0, i-1)

    def update_path_line(self):
        if len(self.selections) > 1:
            xs = [self.coords_2d[i, 0] for i in self.selections]
            ys = [self.coords_2d[i, 1] for i in self.selections]
            self.path_line.setData(xs, ys)

    def search_point(self):
        try:
            idx = int(self.search_input.text())
            if 0 <= idx < self.num_samples:
                x, y = self.coords_2d[idx]
                self.marker_search.setData([x], [y])
                self.plot_widget.setXRange(x-5, x+5, padding=0)
                self.plot_widget.setYRange(y-5, y+5, padding=0)
                self.display_image(self.images[idx], self.image_label, size=180)
                self.active_vector = self.latent_vectors[idx].copy()
                self.update_sliders(self.active_vector)
                self.compute_knn(idx)
                self.statusBar().showMessage(f"Found Point {idx}.")
        except ValueError:
            self.statusBar().showMessage("Invalid ID.")

    def sample_random_z(self):
        z = np.abs(np.random.randn(self.latent_dim) * 0.5)
        z[0:3] *= 100
        z[3] *= 1.5
        
        self.active_vector = z
        img = self.mock_decode(z)
        self.display_image(img, self.image_label, size=180)
        self.update_sliders(z)
        self.tabs.setCurrentIndex(1)
        
        if hasattr(self.reducer, 'transform'):
            coord = self.reducer.transform([z])[0]
        else:
            coord = np.random.uniform(-5, 5, 2)
            
        self.marker_random.setData([coord[0]], [coord[1]])
        self.statusBar().showMessage("Sampled random Z from prior.")

    def generate_pc_traversal(self):
        self.statusBar().showMessage("Computing Principal Direction Traversal...")
        self.clear_layout(self.result_grid_layout)
        
        pca = PCA(n_components=1)
        pca.fit(self.latent_vectors)
        pc1 = pca.components_[0]
        mean_z = pca.mean_
        std = np.std(pca.transform(self.latent_vectors))
        
        steps = 5
        for i in range(steps):
            t = -2 + (4 * i / (steps - 1))
            vec = mean_z + t * std * pc1
            img = self.mock_decode(vec)
            self.add_result_image(img, f"{t:.1f}σ", i)
            
        self.statusBar().showMessage("PC Traversal complete.")
        self.tabs.setCurrentIndex(2)

    def toggle_animation(self):
        if self.animation_timer.isActive():
            self.animation_timer.stop()
            self.btn_play_anim.setText("Play Animation")
        else:
            if len(self.selections) >= 2:
                self.anim_t = 0.0
                self.anim_direction = 1
                self.animation_timer.start(50)
                self.btn_play_anim.setText("Stop Animation")
                self.btn_export_gif.setEnabled(True)
                self.tabs.setCurrentIndex(2)

    def step_animation(self):
        if len(self.selections) < 2:
            self.toggle_animation()
            return
            
        a, b = self.selections[0], self.selections[1]
        vec_a, vec_b = self.latent_vectors[a], self.latent_vectors[b]
        op = self.op_combo.currentText()
        
        self.anim_t += 0.05 * self.anim_direction
        if self.anim_t >= 1.0:
            self.anim_t = 1.0
            self.anim_direction = -1
        elif self.anim_t <= 0.0:
            self.anim_t = 0.0
            self.anim_direction = 1
            
        t = self.anim_t
        
        if op == "Lerp (A -> B)":
            interp_vec = (1 - t) * vec_a + t * vec_b
        else:
            v1n = vec_a / (np.linalg.norm(vec_a) + 1e-8)
            v2n = vec_b / (np.linalg.norm(vec_b) + 1e-8)
            dot = np.clip(np.dot(v1n, v2n), -1.0, 1.0)
            theta = np.arccos(dot)
            sin_theta = np.sin(theta)
            if sin_theta < 1e-6:
                interp_vec = (1 - t) * vec_a + t * vec_b
            else:
                s1 = np.sin((1 - t) * theta) / sin_theta
                s2 = np.sin(t * theta) / sin_theta
                interp_vec = s1 * vec_a + s2 * vec_b
                
        img = self.mock_decode(interp_vec)
        self.display_image(img, self.image_label, size=180)

    def export_gif(self):
        if len(self.selections) < 2: return
        
        file_path, _ = QFileDialog.getSaveFileName(self, "Save GIF", "", "GIF Files (*.gif)")
        if not file_path: return
        
        self.statusBar().showMessage("Generating GIF frames...")
        a, b = self.selections[0], self.selections[1]
        vec_a, vec_b = self.latent_vectors[a], self.latent_vectors[b]
        op = self.op_combo.currentText()
        
        frames = []
        steps = 20
        for i in range(steps):
            t = i / (steps - 1)
            if op == "Lerp (A -> B)":
                interp_vec = (1 - t) * vec_a + t * vec_b
            else:
                v1n = vec_a / (np.linalg.norm(vec_a) + 1e-8)
                v2n = vec_b / (np.linalg.norm(vec_b) + 1e-8)
                dot = np.clip(np.dot(v1n, v2n), -1.0, 1.0)
                theta = np.arccos(dot)
                sin_theta = np.sin(theta)
                if sin_theta < 1e-6:
                    interp_vec = (1 - t) * vec_a + t * vec_b
                else:
                    s1 = np.sin((1 - t) * theta) / sin_theta
                    s2 = np.sin(t * theta) / sin_theta
                    interp_vec = s1 * vec_a + s2 * vec_b
                    
            frames.append(self.mock_decode(interp_vec))
            
        try:
            imageio.mimsave(file_path, frames, duration=0.1)
            self.statusBar().showMessage(f"GIF saved to {file_path}")
        except Exception as e:
            self.statusBar().showMessage(f"Error saving GIF: {e}")

    def process_operation(self):
        op = self.op_combo.currentText()
        self.clear_layout(self.result_grid_layout)
        
        self.traj_line.setData([], [])
        self.traj_scatter.setData([], [])

        if op in ["Lerp (A -> B)", "Slerp (A -> B)"] and len(self.selections) >= 2:
            a, b = self.selections[0], self.selections[1]
            vec_a, vec_b = self.latent_vectors[a], self.latent_vectors[b]
            steps = 5
            traj_coords = []
            
            for i in range(steps):
                t = i / (steps - 1)
                
                if op == "Lerp (A -> B)":
                    interp_vec = (1 - t) * vec_a + t * vec_b
                else:
                    v1n = vec_a / (np.linalg.norm(vec_a) + 1e-8)
                    v2n = vec_b / (np.linalg.norm(vec_b) + 1e-8)
                    dot = np.clip(np.dot(v1n, v2n), -1.0, 1.0)
                    theta = np.arccos(dot)
                    sin_theta = np.sin(theta)
                    if sin_theta < 1e-6:
                        interp_vec = (1 - t) * vec_a + t * vec_b
                    else:
                        s1 = np.sin((1 - t) * theta) / sin_theta
                        s2 = np.sin(t * theta) / sin_theta
                        interp_vec = s1 * vec_a + s2 * vec_b
                        
                img = self.mock_decode(interp_vec)
                self.add_result_image(img, f"t={t:.2f}", i)
                
                if hasattr(self.reducer, 'transform'):
                    coord = self.reducer.transform([interp_vec])[0]
                    traj_coords.append(coord)
                    
            if len(traj_coords) > 0:
                traj_coords = np.array(traj_coords)
                self.traj_line.setData(traj_coords[:, 0], traj_coords[:, 1])
                self.traj_scatter.setData(x=traj_coords[:, 0], y=traj_coords[:, 1])
                
            self.btn_play_anim.setEnabled(True)
            self.tabs.setCurrentIndex(2)

        elif op == "Arithmetic (A + B - C)" and len(self.selections) >= 3:
            a, b, c = self.selections[0], self.selections[1], self.selections[2]
            self.add_result_image(self.images[a], "A", 0)
            self.add_result_image(self.images[b], "B", 1)
            self.add_result_image(self.images[c], "-C", 2)
            vec_result = self.latent_vectors[a] + self.latent_vectors[b] - self.latent_vectors[c]
            img = self.mock_decode(vec_result)
            self.add_result_image(img, "Result", 4)
            self.btn_play_anim.setEnabled(False)
            self.btn_export_gif.setEnabled(False)
            self.tabs.setCurrentIndex(2)

        required_clicks = 3 if "Arithmetic" in op else 2
        if len(self.selections) > required_clicks:
            self.reset_selection()

    def compute_centroid_math(self):
        c1 = self.class_centroids_highd[self.cmb_c1.currentIndex()]
        op1 = self.cmb_op1.currentText()
        c2 = self.class_centroids_highd[self.cmb_c2.currentIndex()]
        op2 = self.cmb_op2.currentText()
        c3 = self.class_centroids_highd[self.cmb_c3.currentIndex()]
        
        if op1 == "+":
            res = c1 + c2
        else:
            res = c1 - c2
            
        if op2 == "+":
            res = res + c3
        else:
            res = res - c3
            
        img = self.mock_decode(res)
        self.clear_layout(self.centroid_result_grid)
        self.add_image_to_grid(img, "Result", self.centroid_result_grid, 0)
        self.statusBar().showMessage("Centroid math computed.")
        
        # Also put it in main viewer
        self.active_vector = res
        self.display_image(img, self.image_label, size=180)
        self.update_sliders(res)

    def add_image_to_grid(self, img_arr, label_text, grid_layout, col_idx):
        h, w, c = img_arr.shape
        q_img = QImage(img_arr.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 0, 0)
        lbl_img = QLabel()
        lbl_img.setPixmap(pixmap.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        lbl_img.setAlignment(Qt.AlignCenter)
        lbl_img.setStyleSheet("background-color: #2d2d2d; border: 1px solid #5a5d61; border-radius: 3px;")
        
        lbl_text = QLabel(label_text)
        lbl_text.setAlignment(Qt.AlignCenter)
        lbl_text.setStyleSheet("font-weight: bold; color: #ffffff;")
        
        vbox.addWidget(lbl_img)
        vbox.addWidget(lbl_text)
        grid_layout.addWidget(container, 0, col_idx)

    def add_result_image(self, img_arr, label_text, col_idx):
        self.add_image_to_grid(img_arr, label_text, self.result_grid_layout, col_idx)
        # Right click to save
        container = self.result_grid_layout.itemAtPosition(0, col_idx).widget()
        lbl_img = container.layout().itemAt(0).widget()
        lbl_img.setContextMenuPolicy(Qt.CustomContextMenu)
        h, w, c = img_arr.shape
        q_img = QImage(img_arr.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        lbl_img.customContextMenuRequested.connect(lambda: self.save_image(pixmap))

    def save_image(self, pixmap):
        if pixmap:
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png)")
            if file_path:
                pixmap.save(file_path)
                self.statusBar().showMessage(f"Saved to {file_path}")

    def clear_layout(self, layout):
        for i in reversed(range(layout.count())):
            widget = layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)

    def display_image(self, img_arr, label_widget, size=150):
        h, w, c = img_arr.shape
        q_img = QImage(img_arr.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        label_widget.setPixmap(pixmap.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def reset_selection(self):
        self.selections = []
        self.hide_markers()
        self.clear_layout(self.result_grid_layout)
        if self.animation_timer.isActive():
            self.toggle_animation()
        self.btn_play_anim.setEnabled(False)
        self.btn_export_gif.setEnabled(False)
        
        op = self.op_combo.currentText()
        if "Arithmetic" in op:
            self.statusBar().showMessage("Select Point A (Then B, then C for A+B-C).")
        else:
            self.statusBar().showMessage("Select Point A (Then B for interpolation).")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    viewer = LatentSpaceViewerProV7()
    viewer.show()
    sys.exit(app.exec())
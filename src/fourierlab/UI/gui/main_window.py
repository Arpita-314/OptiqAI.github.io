import sys
import numpy as np
import torch
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, 
                           QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                           QFileDialog, QGroupBox, QSpinBox, QDoubleSpinBox,
                           QProgressBar, QComboBox, QMessageBox, QTextEdit)
from PyQt5.QtCore import Qt, QThread
from PyQt5.QtGui import QPixmap, QImage
import os
from fourierlab.UI.gui.data_manager import DataManager
from fourierlab.UI.gui.training_manager import TrainingManager
from fourierlab.UI.gui.inverse_design_manager import InverseDesignManager
from fourierlab.core.phase_mask import PhaseMaskGenerator
from fourierlab.core.pattern_generator import PatternGenerator
from fourierlab.UI.gui.automl_manager import AutoMLManager
from fourierlab.core.propagator_factory import PropagatorFactory
from fourierlab.core.quantum_optics_calculator import QuantumOpticsCalculator
from fourierlab.core.quantum_state_generator import QuantumStateGenerator
from fourierlab.core.quantum_operations import QuantumOperations
from fourierlab.core.quantum_state_analyzer import QuantumStateAnalyzer
from fourierlab.core.quantum_state_io import QuantumStateIO
from fourierlab.UI.gui.widgets.workspace_scene import WorkspaceScene
from fourierlab.UI.gui.widgets.workspace_view import WorkspaceView
from fourierlab.UI.gui.widgets.lens_item import LensItem
import requests
import json

class TrainingThread(QThread):
    def __init__(self, training_manager, train_loader, val_loader, epochs):
        super().__init__()
        self.training_manager = training_manager
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
    
    def run(self):
        self.training_manager.train(self.train_loader, self.val_loader, self.epochs)

class GenerationThread(QThread):
    def __init__(self, manager, target, wavelength, pixel_size, iterations):
        super().__init__()
        self.manager = manager
        self.target = target
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.iterations = iterations
    
    def run(self):
        results = self.manager.generate_phase_mask(
            self.target,
            self.wavelength,
            self.pixel_size,
            self.iterations
        )
        if results:
            self.manager.generation_complete.emit(results)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fourier Optics AutoML for Photonics")
        self.setMinimumSize(1200, 800)
        self.backend_url = "http://127.0.0.1:8000"  # Replace with your backend URL

        # --- Application Stylesheet ---
        self.setStyleSheet("""
            /* --- General Styles --- */
            QMainWindow {
                background-color: #E8F0FE; /* Light blue background */
                font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
                font-size: 14px;
            }

            QGroupBox {
                font-weight: bold;
                border: 1px solid #7EA6E0; /* Light blue border */
                border-radius: 5px;
                margin-top: 1em;
                padding: 10px;
                background-color: #FFFFFF; /* White group box background */
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #3C64B1; /* Dark blue title */
            }

            QPushButton {
                padding: 8px 20px;
                background-color: #5CB85C; /* Green button */
                color: white;
                border: none;
                border-radius: 5px;
                font-weight: bold;
                min-width: 120px;
            }

            QPushButton:hover {
                background-color: #4CAE4C; /* Darker green on hover */
            }

            QPushButton:pressed {
                background-color: #398439; /* Even darker green when pressed */
            }

            QPushButton:disabled {
                background-color: #D9D9D9; /* Light gray disabled button */
                color: #8C8C8C; /* Dark gray text */
            }

            QLabel {
                padding: 5px;
                color: #333333; /* Dark gray text */
            }

            QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
                padding: 6px;
                border: 1px solid #A6CAE0; /* Light blue border */
                border-radius: 4px;
                background-color: #F0F8FF; /* Light blue input fields */
            }

            QProgressBar {
                border: 1px solid #A6CAE0; /* Light blue border */
                border-radius: 5px;
                text-align: center;
                background-color: #F0F0F0; /* Light gray background */
                color: #333333; /* Dark gray text */
            }

            QProgressBar::chunk {
                background-color: #5BC0DE; /* Cyan progress bar */
                border-radius: 5px;
            }

            QTextEdit {
                border: 1px solid #A6CAE0; /* Light blue border */
                border-radius: 4px;
                background-color: #F0F8FF; /* Light blue input fields */
                color: #333333; /* Dark gray text */
            }

            QTabWidget::pane { /* The tab widget frame */
                border-top: 2px solid #A6CAE0; /* Light blue border */
            }

            QTabWidget::tab-bar {
                left: 5px; /* Move the tab bar to the right to add some visual space */
            }

            QTabBar::tab {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                            stop: 0 #E0EAF5, stop: 0.4 #D0D9E8,
                                            stop: 0.5 #C0C9DA, stop: 1.0 #B0B8CC);
                border: 2px solid #A6CAE0; /* Light blue border */
                border-bottom-color: #A6CAE0; /* same as the pane color */
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                min-width: 8ex;
                padding: 5px;
            }

            QTabBar::tab:selected, QTabBar::tab:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                            stop: 0 #F0F5FA, stop: 0.4 #E0E9F4,
                                            stop: 0.5 #D0D9E3, stop: 1.0 #C0C8D2);
            }

            QTabBar::tab:selected {
                border-color: #7EA6E0; /* Light blue border */
                border-bottom-color: #A6CAE0; /* same as pane color */
            }

            QTabBar::tab:!selected {
                margin-top: 2px; /* make non-selected tabs look smaller */
            }
        """)

        # Initialize device selection
        self.device_mode = 'auto'  # Default to auto
        self.check_device_capabilities()

        # Initialize managers
        self.data_manager = DataManager()
        self.data_manager.data_loaded.connect(self.on_data_loaded)
        self.data_manager.preprocessing_complete.connect(self.on_preprocessing_complete)

        self.training_manager = TrainingManager()
        self.training_manager.training_progress.connect(self.on_training_progress)
        self.training_manager.training_complete.connect(self.on_training_complete)
        self.training_manager.training_error.connect(self.on_training_error)
        
        self.inverse_design_manager = InverseDesignManager()
        self.inverse_design_manager.progress_updated.connect(self.update_generation_progress)
        self.inverse_design_manager.generation_complete.connect(self.on_generation_complete)
        self.inverse_design_manager.error_occurred.connect(self.on_generation_error)
        
        self.automl_manager = AutoMLManager()
        self.automl_manager.trial_complete.connect(self.on_trial_complete)
        self.automl_manager.optimization_complete.connect(self.on_optimization_complete)
        self.automl_manager.error_occurred.connect(self.on_automl_error)
        
        # Initialize quantum optics components
        self.quantum_calculator = QuantumOpticsCalculator()
        self.quantum_generator = QuantumStateGenerator(self.quantum_calculator)
        self.quantum_operations = QuantumOperations(self.quantum_calculator)
        self.quantum_analyzer = QuantumStateAnalyzer(self.quantum_calculator)
        self.quantum_io = QuantumStateIO(self.quantum_calculator)
        
        # Setup UI
        self.tabs = QTabWidget()
        self.tabs.addTab(self.data_analysis_tab(), "Data Analysis")
        self.tabs.addTab(self.inverse_design_tab(), "Inverse Design")
        self.tabs.addTab(self.automl_tab(), "AutoML")
        self.tabs.addTab(self.dataset_generation_tab(), "Dataset Generation")
        self.tabs.addTab(self.fourier_optics_tab(), "Fourier Optics")
        self.setCentralWidget(self.tabs)
        
    def check_device_capabilities(self):
        """Check device capabilities and set appropriate mode."""
        try:
            if torch.cuda.is_available():
                self.device_mode = 'gpu'
                self.gpu_name = torch.cuda.get_device_name(0)
                self.gpu_memory = torch.cuda.get_device_properties(0).total_memory
            else:
                self.device_mode = 'cpu'
                self.gpu_name = None
                self.gpu_memory = None
        except Exception:
            self.device_mode = 'cpu'
            self.gpu_name = None
            self.gpu_memory = None

    def data_analysis_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Import Section
        import_group = QGroupBox("1. Import Data")
        import_layout = QVBoxLayout()
        self.import_btn = QPushButton("Select Data Directory")
        self.import_btn.clicked.connect(self.select_data_directory)
        import_layout.addWidget(self.import_btn)
        self.import_status = QLabel("No data loaded")
        import_layout.addWidget(self.import_status)
        import_group.setLayout(import_layout)
        layout.addWidget(import_group)
        
        # Preprocessing Section
        preprocess_group = QGroupBox("2. Preprocessing")
        preprocess_layout = QVBoxLayout()
        
        # Target size controls
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Target Size:"))
        self.width_spin = QSpinBox()
        self.width_spin.setRange(64, 1024)
        self.width_spin.setValue(256)
        self.width_spin.setSingleStep(32)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(64, 1024)
        self.height_spin.setValue(256)
        self.height_spin.setSingleStep(32)
        size_layout.addWidget(self.width_spin)
        size_layout.addWidget(QLabel("x"))
        size_layout.addWidget(self.height_spin)
        preprocess_layout.addLayout(size_layout)
        
        self.preprocess_btn = QPushButton("Run Preprocessing")
        self.preprocess_btn.clicked.connect(self.run_preprocessing)
        self.preprocess_btn.setEnabled(False)  # Disabled until data is loaded
        preprocess_layout.addWidget(self.preprocess_btn)
        self.preprocess_status = QLabel("")
        preprocess_layout.addWidget(self.preprocess_status)
        preprocess_group.setLayout(preprocess_layout)
        layout.addWidget(preprocess_group)
        
        # Training Section
        training_group = QGroupBox("3. Training")
        training_layout = QVBoxLayout()
        
        # Training parameters
        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["FourierCNN", "AutoML Search"])
        params_layout.addWidget(self.model_combo)
        
        params_layout.addWidget(QLabel("Epochs:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(10)
        params_layout.addWidget(self.epochs_spin)
        training_layout.addLayout(params_layout)
        
        self.train_btn = QPushButton("Start Training")
        self.train_btn.clicked.connect(self.start_training)
        self.train_btn.setEnabled(False)  # Disabled until preprocessing is done
        training_layout.addWidget(self.train_btn)
        
        self.progress_bar = QProgressBar()
        training_layout.addWidget(self.progress_bar)
        
        self.training_status = QLabel("")
        training_layout.addWidget(self.training_status)
        
        training_group.setLayout(training_layout)
        layout.addWidget(training_group)
        
        # Results Section
        results_group = QGroupBox("4. Results")
        results_layout = QVBoxLayout()
        self.results_label = QLabel("Training metrics and visualizations will appear here")
        results_layout.addWidget(self.results_label)
        
        self.save_model_btn = QPushButton("Save Model")
        self.save_model_btn.clicked.connect(self.save_model)
        self.save_model_btn.setEnabled(False)
        results_layout.addWidget(self.save_model_btn)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        tab.setLayout(layout)
        return tab
        
    def select_data_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Data Directory")
        if dir_path:
            try:
                if self.data_manager.load_data(dir_path):
                    self.import_status.setText(f"Loaded data from: {dir_path}")
                    self.preprocess_btn.setEnabled(True)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")

    def on_data_loaded(self, message):
        self.import_status.setText(message)

    def run_preprocessing(self):
        try:
            target_size = (self.width_spin.value(), self.height_spin.value())
            self.preprocess_status.setText("Preprocessing in progress...")
            self.preprocess_btn.setEnabled(False)
            self.data_manager.preprocess_data(target_size)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Preprocessing failed: {str(e)}")
            self.preprocess_btn.setEnabled(True)

    def on_preprocessing_complete(self, result):
        if isinstance(result, str):  # Error message
            self.preprocess_status.setText(result)
            self.preprocess_btn.setEnabled(True)
        else:  # Success
            self.preprocess_status.setText("Preprocessing complete!")
            self.train_btn.setEnabled(True)
            self.processed_data = result

    def start_training(self):
        try:
            # Setup training
            train_loader, val_loader = self.training_manager.setup_training(
                self.processed_data,
                model_type=self.model_combo.currentText(),
                epochs=self.epochs_spin.value()
            )
            
            if train_loader is None or val_loader is None:
                return
            
            # Disable training controls
            self.train_btn.setEnabled(False)
            self.model_combo.setEnabled(False)
            self.epochs_spin.setEnabled(False)
            
            # Start training in a separate thread
            self.training_thread = TrainingThread(
                self.training_manager,
                train_loader,
                val_loader,
                self.epochs_spin.value()
            )
            self.training_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start training: {str(e)}")
            self.train_btn.setEnabled(True)

    def on_training_progress(self, progress):
        self.progress_bar.setValue(progress)

    def on_training_complete(self, results):
        # Update results display
        self.training_status.setText(
            f"Epoch {results['epoch']}: "
            f"Val Loss = {results['val_loss']:.4f}, "
            f"Phase RMSE = {results['phase_rmse']:.4f}"
        )
        
        # Enable save button after training
        self.save_model_btn.setEnabled(True)
        
        # Re-enable training controls
        self.train_btn.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.epochs_spin.setEnabled(True)

    def on_training_error(self, error):
        QMessageBox.critical(self, "Training Error", error)
        self.train_btn.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.epochs_spin.setEnabled(True)

    def save_model(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Model",
            "",
            "PyTorch Models (*.pt);;All Files (*.*)"
        )
        
        if file_path:
            if self.training_manager.save_model(file_path):
                QMessageBox.information(self, "Success", "Model saved successfully!")
            else:
                QMessageBox.critical(self, "Error", "Failed to save model!")

    def inverse_design_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Device Selection Section
        device_group = QGroupBox("1. Device Selection")
        device_layout = QVBoxLayout()
        
        # Device mode selection
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Device Mode:"))
        self.device_mode_combo = QComboBox()
        self.device_mode_combo.addItems(["Auto", "GPU", "CPU"])
        self.device_mode_combo.setCurrentText("Auto")
        self.device_mode_combo.currentTextChanged.connect(self.on_device_mode_changed)
        mode_layout.addWidget(self.device_mode_combo)
        device_layout.addLayout(mode_layout)
        
        # Device info
        self.device_info = QLabel()
        self.update_device_info()
        device_layout.addWidget(self.device_info)
        
        # Feature info
        self.feature_info = QLabel()
        self.update_feature_info()
        device_layout.addWidget(self.feature_info)
        
        device_group.setLayout(device_layout)
        layout.addWidget(device_group)
        
        # Target Specification Section
        target_group = QGroupBox("2. Specify Target")
        target_layout = QVBoxLayout()
        
        # Target type selection
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Target Type:"))
        self.target_type = QComboBox()
        self.target_type.addItems(["Image", "Pattern"])
        self.target_type.currentTextChanged.connect(self.on_target_type_changed)
        type_layout.addWidget(self.target_type)
        target_layout.addLayout(type_layout)
        
        # Target image/pattern controls
        self.target_controls = QWidget()
        target_controls_layout = QVBoxLayout()
        
        # Image upload button
        self.upload_btn = QPushButton("Upload Target Image")
        self.upload_btn.clicked.connect(self.upload_target_image)
        target_controls_layout.addWidget(self.upload_btn)
        
        # Pattern parameters
        pattern_params = QHBoxLayout()
        pattern_params.addWidget(QLabel("Pattern Size:"))
        self.pattern_size = QSpinBox()
        self.pattern_size.setRange(64, 1024)
        self.pattern_size.setValue(256)
        self.pattern_size.setSingleStep(32)
        pattern_params.addWidget(self.pattern_size)
        
        pattern_params.addWidget(QLabel("Pattern Type:"))
        self.pattern_type = QComboBox()
        self.pattern_type.addItems(["cross", "circle", "square", "grating", "spiral"])
        self.pattern_type.currentTextChanged.connect(self.update_pattern_preview)
        pattern_params.addWidget(self.pattern_type)
        
        target_controls_layout.addLayout(pattern_params)
        
        # Pattern-specific parameters
        self.pattern_params = QWidget()
        pattern_params_layout = QVBoxLayout()
        
        # Width parameter
        width_layout = QHBoxLayout()
        width_layout.addWidget(QLabel("Width:"))
        self.pattern_width = QSpinBox()
        self.pattern_width.setRange(1, 100)
        self.pattern_width.setValue(25)
        self.pattern_width.valueChanged.connect(self.update_pattern_preview)
        width_layout.addWidget(self.pattern_width)
        pattern_params_layout.addLayout(width_layout)
        
        # Frequency parameter
        freq_layout = QHBoxLayout()
        freq_layout.addWidget(QLabel("Frequency:"))
        self.pattern_freq = QDoubleSpinBox()
        self.pattern_freq.setRange(1, 50)
        self.pattern_freq.setValue(10)
        self.pattern_freq.setSingleStep(0.5)
        self.pattern_freq.valueChanged.connect(self.update_pattern_preview)
        freq_layout.addWidget(self.pattern_freq)
        pattern_params_layout.addLayout(freq_layout)
        
        self.pattern_params.setLayout(pattern_params_layout)
        target_controls_layout.addWidget(self.pattern_params)
        
        self.target_controls.setLayout(target_controls_layout)
        target_layout.addWidget(self.target_controls)
        
        # Target preview
        self.target_preview = QLabel("Target preview will appear here")
        self.target_preview.setAlignment(Qt.AlignCenter)
        self.target_preview.setMinimumSize(256, 256)
        self.target_preview.setStyleSheet("border: 1px solid #ccc;")
        target_layout.addWidget(self.target_preview)
        
        target_group.setLayout(target_layout)
        layout.addWidget(target_group)
        
        # Generation Parameters Section
        params_group = QGroupBox("3. Generation Parameters")
        params_layout = QVBoxLayout()
        
        # Wavelength
        wavelength_layout = QHBoxLayout()
        wavelength_layout.addWidget(QLabel("Wavelength (nm):"))
        self.wavelength = QDoubleSpinBox()
        self.wavelength.setRange(200, 2000)
        self.wavelength.setValue(632.8)
        self.wavelength.setSingleStep(0.1)
        wavelength_layout.addWidget(self.wavelength)
        params_layout.addLayout(wavelength_layout)
        
        # Pixel size
        pixel_layout = QHBoxLayout()
        pixel_layout.addWidget(QLabel("Pixel Size (μm):"))
        self.pixel_size = QDoubleSpinBox()
        self.pixel_size.setRange(0.1, 10.0)
        self.pixel_size.setValue(5.0)
        self.pixel_size.setSingleStep(0.1)
        pixel_layout.addWidget(self.pixel_size)
        params_layout.addLayout(pixel_layout)
        
        # Optimization parameters
        opt_group = QGroupBox("Optimization Parameters")
        opt_layout = QVBoxLayout()
        
        # Iterations
        iter_layout = QHBoxLayout()
        iter_layout.addWidget(QLabel("Iterations:"))
        self.iterations = QSpinBox()
        self.iterations.setRange(100, 10000)
        self.iterations.setValue(1000)
        self.iterations.setSingleStep(100)
        iter_layout.addWidget(self.iterations)
        opt_layout.addLayout(iter_layout)
        
        # Learning rate
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("Learning Rate:"))
        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setRange(0.0001, 0.1)
        self.learning_rate.setValue(0.01)
        self.learning_rate.setSingleStep(0.001)
        lr_layout.addWidget(self.learning_rate)
        opt_layout.addLayout(lr_layout)
        
        # Optimizer type
        opt_type_layout = QHBoxLayout()
        opt_type_layout.addWidget(QLabel("Optimizer:"))
        self.optimizer_type = QComboBox()
        self.optimizer_type.addItems(["adam", "sgd"])
        opt_type_layout.addWidget(self.optimizer_type)
        opt_layout.addLayout(opt_type_layout)
        
        # Loss weights
        weights_layout = QHBoxLayout()
        weights_layout.addWidget(QLabel("Smoothness Weight:"))
        self.smoothness_weight = QDoubleSpinBox()
        self.smoothness_weight.setRange(0, 1)
        self.smoothness_weight.setValue(0.1)
        self.smoothness_weight.setSingleStep(0.01)
        weights_layout.addWidget(self.smoothness_weight)
        
        weights_layout.addWidget(QLabel("Contrast Weight:"))
        self.contrast_weight = QDoubleSpinBox()
        self.contrast_weight.setRange(0, 1)
        self.contrast_weight.setValue(0.05)
        self.contrast_weight.setSingleStep(0.01)
        weights_layout.addWidget(self.contrast_weight)
        opt_layout.addLayout(weights_layout)
        
        opt_group.setLayout(opt_layout)
        params_layout.addWidget(opt_group)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Generation Section
        generate_group = QGroupBox("4. Generate Phase Mask")
        generate_layout = QVBoxLayout()
        
        self.generate_btn = QPushButton("Generate Phase Mask")
        self.generate_btn.clicked.connect(self.generate_phase_mask)
        generate_layout.addWidget(self.generate_btn)
        
        self.generation_progress = QProgressBar()
        generate_layout.addWidget(self.generation_progress)
        
        # Metrics display
        self.metrics_label = QLabel("")
        generate_layout.addWidget(self.metrics_label)
        
        self.generation_status = QLabel("")
        generate_layout.addWidget(self.generation_status)
        
        generate_group.setLayout(generate_layout)
        layout.addWidget(generate_group)
        
        # Results Section
        results_group = QGroupBox("5. Results")
        results_layout = QVBoxLayout()
        
        # Results display
        results_display = QHBoxLayout()
        
        # Phase mask preview
        self.phase_preview = QLabel("Phase mask will appear here")
        self.phase_preview.setAlignment(Qt.AlignCenter)
        self.phase_preview.setMinimumSize(256, 256)
        self.phase_preview.setStyleSheet("border: 1px solid #ccc;")
        results_display.addWidget(self.phase_preview)
        
        # Simulated output preview
        self.output_preview = QLabel("Simulated output will appear here")
        self.output_preview.setAlignment(Qt.AlignCenter)
        self.output_preview.setMinimumSize(256, 256)
        self.output_preview.setStyleSheet("border: 1px solid #ccc;")
        results_display.addWidget(self.output_preview)
        
        results_layout.addLayout(results_display)
        
        # Save controls
        save_layout = QHBoxLayout()
        self.save_phase_btn = QPushButton("Save Phase Mask")
        self.save_phase_btn.clicked.connect(self.save_phase_mask)
        self.save_phase_btn.setEnabled(False)
        save_layout.addWidget(self.save_phase_btn)
        
        self.save_output_btn = QPushButton("Save Output")
        self.save_output_btn.clicked.connect(self.save_output)
        self.save_output_btn.setEnabled(False)
        save_layout.addWidget(self.save_output_btn)
        
        results_layout.addLayout(save_layout)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        tab.setLayout(layout)
        return tab
    
    def on_target_type_changed(self, target_type):
        """Handle target type selection change"""
        if target_type == "Image":
            self.upload_btn.setEnabled(True)
            self.pattern_size.setEnabled(False)
            self.pattern_type.setEnabled(False)
            self.pattern_width.setEnabled(False)
            self.pattern_freq.setEnabled(False)
        else:  # Pattern
            self.upload_btn.setEnabled(False)
            self.pattern_size.setEnabled(True)
            self.pattern_type.setEnabled(True)
            self.pattern_width.setEnabled(True)
            self.pattern_freq.setEnabled(True)
            self.update_pattern_preview()
    
    def upload_target_image(self):
        """Handle target image upload"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Target Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.tif *.tiff);;All Files (*.*)"
        )
        
        if file_path:
            try:
                # Load and display the image
                pixmap = QPixmap(file_path)
                scaled_pixmap = pixmap.scaled(
                    self.target_preview.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.target_preview.setPixmap(scaled_pixmap)
                self.target_image_path = file_path
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")
    
    def update_pattern_preview(self):
        """Update the pattern preview"""
        if self.target_type.currentText() == "Pattern":
            # Get pattern parameters
            size = (self.pattern_size.value(), self.pattern_size.value())
            pattern_type = self.pattern_type.currentText()
            params = {
                'width': self.pattern_width.value(),
                'frequency': self.pattern_freq.value()
            }
            
            # Generate and display pattern
            pattern_gen = PatternGenerator()
            pattern = pattern_gen.generate_pattern(
                pattern_type='vortex',
                size=size,
                wavelength=632.8e-9,
                pixel_size=5e-6,
                order=2  # For vortex
            )
            
            if pattern is not None:
                self._display_target(pattern)
    
    def _display_target(self, target):
        """Display target pattern/image"""
        if target is None:
            return
            
        # Convert tensor to numpy array if needed
        if torch.is_tensor(target):
            target = target.detach().cpu().numpy()
            
        # Convert to uint8 for display
        target_np = (target * 255).astype(np.uint8)
        
        # Convert to QImage and display
        height, width = target_np.shape
        bytes_per_line = width
        image = QImage(target_np.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        self.target_preview.setPixmap(QPixmap.fromImage(image).scaled(
            self.target_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
    
    def generate_phase_mask(self):
        """Generate phase mask by calling the backend API."""
        try:
            # Get parameters from GUI
            params = {
                'wavelength': self.wavelength.value() * 1e-9,
                'pixel_size': self.pixel_size.value() * 1e-6,
                'iterations': self.iterations.value(),
                'learning_rate': self.learning_rate.value(),
                'optimizer_type': self.optimizer_type.currentText(),
                'smoothness_weight': self.smoothness_weight.value(),
                'contrast_weight': self.contrast_weight.value()
            }

            # Call the backend API
            url = f"{self.backend_url}/simulate"
            headers = {'Content-Type': 'application/json'}
            data = {'simulation_type': 'FDTD', 'params': params}  # Or 'RAY_TRACING'
            response = requests.post(url, headers=headers, data=json.dumps(data))

            # Process the response
            if response.status_code == 200:
                result = response.json()['result']
                # Display the result in the GUI
                self.generation_status.setText(f"Simulation Result: {result}")
            else:
                QMessageBox.critical(self, "Error", f"Backend Error: {response.text}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate phase mask: {str(e)}")
    
    def update_generation_progress(self, iteration, metrics):
        """Update progress bar and metrics display."""
        self.generation_progress.setValue(iteration)
        metrics_text = "\n".join([
            f"{key}: {value:.4f}" for key, value in metrics.items()
        ])
        self.metrics_label.setText(metrics_text)
    
    def on_generation_complete(self, phase_mask, simulated_output):
        """Handle generation completion."""
        try:
            # Display results
            self._display_phase_mask(phase_mask)
            self._display_simulated_output(simulated_output)
            
            # Enable save buttons
            self.save_phase_btn.setEnabled(True)
            self.save_output_btn.setEnabled(True)
            
            # Update status
            self.generation_status.setText("Generation complete!")
            
            # Clean up thread
            self.generation_thread.quit()
            self.generation_thread.wait()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to display results: {str(e)}")
        finally:
            self.generate_btn.setEnabled(True)
    
    def on_generation_error(self, error_msg):
        """Handle generation error."""
        QMessageBox.critical(self, "Error", f"Generation failed: {error_msg}")
        self.generate_btn.setEnabled(True)
        self.generation_thread.quit()
        self.generation_thread.wait()
    
    def _display_phase_mask(self, phase_mask):
        """Display phase mask in preview."""
        # Convert phase mask to image
        phase_img = self._tensor_to_qimage(phase_mask)
        
        # Scale to preview size
        scaled_img = phase_img.scaled(
            self.phase_preview.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        self.phase_preview.setPixmap(QPixmap.fromImage(scaled_img))
    
    def _display_simulated_output(self, output):
        """Display simulated output in preview."""
        # Convert output to image
        output_img = self._tensor_to_qimage(output)
        
        # Scale to preview size
        scaled_img = output_img.scaled(
            self.output_preview.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        self.output_preview.setPixmap(QPixmap.fromImage(scaled_img))
    
    def save_phase_mask(self):
        """Save generated phase mask to file."""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Phase Mask",
                "",
                "PNG Files (*.png);;All Files (*)"
            )
            
            if file_path:
                # Get current phase mask from preview
                phase_mask = self.phase_preview.pixmap().toImage()
                phase_mask.save(file_path)
                QMessageBox.information(self, "Success", "Phase mask saved successfully!")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save phase mask: {str(e)}")
    
    def save_output(self):
        """Save simulated output to file."""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Simulated Output",
                "",
                "PNG Files (*.png);;All Files (*)"
            )
            
            if file_path:
                # Get current output from preview
                output = self.output_preview.pixmap().toImage()
                output.save(file_path)
                QMessageBox.information(self, "Success", "Simulated output saved successfully!")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save output: {str(e)}")
    
    def _tensor_to_qimage(self, tensor):
        """Convert PyTorch tensor to QImage."""
        # Convert to numpy array
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()
        
        # Normalize to 0-255 range
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        tensor = (tensor * 255).astype(np.uint8)
        
        # Convert to QImage
        height, width = tensor.shape
        bytes_per_line = width
        return QImage(tensor.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

    def on_device_mode_changed(self, mode):
        """Handle device mode change."""
        self.device_mode = mode.lower()
        self.update_device_info()
        self.update_feature_info()
        
        # Update propagator if it exists
        if hasattr(self, 'phase_generator'):
            self.phase_generator = PropagatorFactory.create_propagator(
                mode=self.device_mode,
                device=0 if self.device_mode == 'gpu' else None
            )
    
    def update_device_info(self):
        """Update device information display."""
        if self.device_mode == 'gpu' and self.gpu_name:
            info = f"GPU: {self.gpu_name}\n"
            if self.gpu_memory:
                info += f"Memory: {self.gpu_memory / 1024**3:.1f} GB"
        else:
            info = "Using CPU mode"
        self.device_info.setText(info)
    
    def update_feature_info(self):
        """Update feature information display."""
        features = PropagatorFactory.get_available_features(self.device_mode)
        settings = PropagatorFactory.get_recommended_settings(self.device_mode)
        
        info = "Available Features:\n"
        info += f"• Max Field Size: {features['max_field_size']}x{features['max_field_size']}\n"
        info += f"• Batch Processing: {'Yes' if features['batch_processing'] else 'No'}\n"
        info += f"• Mixed Precision: {'Yes' if features['mixed_precision'] else 'No'}\n"
        info += f"• Multi-GPU: {'Yes' if features['multi_gpu'] else 'No'}\n"
        info += f"• Advanced Features: {'Yes' if features['advanced_features'] else 'No'}\n"
        info += f"\nRecommended Settings:\n"
        info += f"• Field Size: {settings['field_size']}x{settings['field_size']}\n"
        info += f"• Batch Size: {settings['batch_size']}\n"
        info += f"• Precision: {settings['precision']}\n"
        info += f"• Memory Limit: {settings['memory_limit']*100}%\n"
        info += f"• Optimization: {settings['optimization_level']}"
        
        self.feature_info.setText(info)

    def automl_tab(self):
        """Create the AutoML tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Data Section
        data_group = QGroupBox("1. Data")
        data_layout = QVBoxLayout()
        
        # Data selection
        data_select = QHBoxLayout()
        data_select.addWidget(QLabel("Dataset:"))
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["Training Data", "Custom Dataset"])
        data_select.addWidget(self.dataset_combo)
        data_layout.addLayout(data_select)
        
        # Custom dataset controls
        self.custom_data_controls = QWidget()
        custom_data_layout = QVBoxLayout()
        
        self.load_data_btn = QPushButton("Load Custom Dataset")
        self.load_data_btn.clicked.connect(self.load_custom_dataset)
        custom_data_layout.addWidget(self.load_data_btn)
        
        self.custom_data_controls.setLayout(custom_data_layout)
        data_layout.addWidget(self.custom_data_controls)
        
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        
        # Optimization Settings
        settings_group = QGroupBox("2. Optimization Settings")
        settings_layout = QVBoxLayout()
        
        # Number of trials
        trials_layout = QHBoxLayout()
        trials_layout.addWidget(QLabel("Number of Trials:"))
        self.n_trials = QSpinBox()
        self.n_trials.setRange(10, 200)
        self.n_trials.setValue(50)
        trials_layout.addWidget(self.n_trials)
        settings_layout.addLayout(trials_layout)
        
        # Search space
        search_group = QGroupBox("Search Space")
        search_layout = QVBoxLayout()
        
        # Model architecture
        arch_layout = QHBoxLayout()
        arch_layout.addWidget(QLabel("Min/Max Conv Layers:"))
        self.min_layers = QSpinBox()
        self.min_layers.setRange(2, 5)
        self.min_layers.setValue(2)
        self.max_layers = QSpinBox()
        self.max_layers.setRange(2, 5)
        self.max_layers.setValue(5)
        arch_layout.addWidget(self.min_layers)
        arch_layout.addWidget(QLabel("-"))
        arch_layout.addWidget(self.max_layers)
        search_layout.addLayout(arch_layout)
        
        # Filter range
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter Range:"))
        self.min_filters = QSpinBox()
        self.min_filters.setRange(16, 64)
        self.min_filters.setValue(16)
        self.max_filters = QSpinBox()
        self.max_filters.setRange(64, 256)
        self.max_filters.setValue(128)
        filter_layout.addWidget(self.min_filters)
        filter_layout.addWidget(QLabel("-"))
        filter_layout.addWidget(self.max_filters)
        search_layout.addLayout(filter_layout)
        
        search_group.setLayout(search_layout)
        settings_layout.addWidget(search_group)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Optimization Controls
        control_group = QGroupBox("3. Optimization")
        control_layout = QVBoxLayout()
        
        self.start_optimization_btn = QPushButton("Start Optimization")
        self.start_optimization_btn.clicked.connect(self.start_optimization)
        control_layout.addWidget(self.start_optimization_btn)
        
        self.optimization_progress = QProgressBar()
        control_layout.addWidget(self.optimization_progress)
        
        self.optimization_status = QLabel("")
        control_layout.addWidget(self.optimization_status)
        
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # Results Section
        results_group = QGroupBox("4. Results")
        results_layout = QVBoxLayout()
        
        # Best model info
        self.best_model_info = QLabel("No optimization results yet")
        results_layout.addWidget(self.best_model_info)
        
        # Trial history
        self.trial_history = QTextEdit()
        self.trial_history.setReadOnly(True)
        results_layout.addWidget(self.trial_history)
        
        # Save controls
        save_layout = QHBoxLayout()
        self.save_model_btn = QPushButton("Save Best Model")
        self.save_model_btn.clicked.connect(self.save_best_model)
        self.save_model_btn.setEnabled(False)
        save_layout.addWidget(self.save_model_btn)
        
        self.save_report_btn = QPushButton("Save Report")
        self.save_report_btn.clicked.connect(self.save_optimization_report)
        self.save_report_btn.setEnabled(False)
        save_layout.addWidget(self.save_report_btn)
        
        results_layout.addLayout(save_layout)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        tab.setLayout(layout)
        return tab
    
    def load_custom_dataset(self):
        """Load custom dataset for AutoML"""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Dataset Directory")
        if dir_path:
            try:
                # Load dataset
                train_loader, val_loader = self.data_manager.load_dataset(dir_path)
                if train_loader and val_loader:
                    self.train_loader = train_loader
                    self.val_loader = val_loader
                    self.optimization_status.setText("Dataset loaded successfully")
                    self.start_optimization_btn.setEnabled(True)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load dataset: {str(e)}")
    
    def start_optimization(self):
        """Start AutoML optimization"""
        try:
            # Get dataset
            if self.dataset_combo.currentText() == "Training Data":
                if not hasattr(self, 'processed_data'):
                    QMessageBox.warning(self, "Warning", "Please load and preprocess data first")
                    return
                train_loader, val_loader = self.training_manager.setup_training(
                    self.processed_data,
                    model_type="AutoML",
                    epochs=1  # Will be optimized by AutoML
                )
            else:
                if not hasattr(self, 'train_loader'):
                    QMessageBox.warning(self, "Warning", "Please load custom dataset first")
                    return
                train_loader, val_loader = self.train_loader, self.val_loader
            
            # Disable controls
            self.start_optimization_btn.setEnabled(False)
            self.optimization_progress.setRange(0, self.n_trials.value())
            self.optimization_progress.setValue(0)
            
            # Start optimization
            self.automl_manager.optimize(
                train_loader,
                val_loader,
                n_trials=self.n_trials.value()
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start optimization: {str(e)}")
            self.start_optimization_btn.setEnabled(True)
    
    def on_trial_complete(self, trial_results):
        """Handle trial completion"""
        self.optimization_progress.setValue(trial_results['number'])
        self.trial_history.append(
            f"Trial {trial_results['number']}: "
            f"Value = {trial_results['value']:.4f}\n"
            f"Params: {trial_results['params']}\n"
        )
    
    def on_optimization_complete(self, results):
        """Handle optimization completion"""
        # Update best model info
        self.best_model_info.setText(
            f"Best Model:\n"
            f"Validation Loss: {results['best_value']:.4f}\n"
            f"Parameters: {results['best_params']}\n"
            f"Model Architecture:\n{results['model_summary']}"
        )
        
        # Enable save buttons
        self.save_model_btn.setEnabled(True)
        self.save_report_btn.setEnabled(True)
        
        # Update status
        self.optimization_status.setText("Optimization complete!")
        
        # Re-enable start button
        self.start_optimization_btn.setEnabled(True)
    
    def on_automl_error(self, error_msg):
        """Handle AutoML error"""
        QMessageBox.critical(self, "Error", f"AutoML error: {error_msg}")
        self.start_optimization_btn.setEnabled(True)
    
    def save_best_model(self):
        """Save the best model found during optimization"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Best Model",
            "",
            "PyTorch Models (*.pt);;All Files (*.*)"
        )
        
        if file_path:
            try:
                model = self.automl_manager.get_best_model()
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'params': self.automl_manager.get_best_params()
                }, file_path)
                QMessageBox.information(self, "Success", "Model saved successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save model: {str(e)}")
    
    def save_optimization_report(self):
        """Save optimization report"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Optimization Report",
            "",
            "Text Files (*.txt);;All Files (*.*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write("AutoML Optimization Report\n")
                    f.write("=======================\n\n")
                    
                    # Write best model info
                    f.write("Best Model:\n")
                    f.write(f"Validation Loss: {self.automl_manager.study.best_value:.4f}\n")
                    f.write(f"Parameters: {self.automl_manager.best_params}\n\n")
                    
                    # Write trial history
                    f.write("Trial History:\n")
                    for trial in self.automl_manager.get_trial_history():
                        f.write(f"Trial {trial['number']}:\n")
                        f.write(f"Value: {trial['value']:.4f}\n")
                        f.write(f"Parameters: {trial['params']}\n")
                        f.write(f"State: {trial['state']}\n\n")
                
                QMessageBox.information(self, "Success", "Report saved successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save report: {str(e)}")

    def dataset_generation_tab(self):
        """Create the Dataset Generation tab."""
        tab = QWidget()
        layout = QVBoxLayout()

        # Parameter Ranges Section
        param_group = QGroupBox("1. Parameter Ranges")
        param_layout = QVBoxLayout()

        # Wavelength range
        wavelength_layout = QHBoxLayout()
        wavelength_layout.addWidget(QLabel("Wavelength Range (nm):"))
        self.wavelength_min = QDoubleSpinBox()
        self.wavelength_min.setRange(200, 2000)
        self.wavelength_min.setValue(400)
        self.wavelength_min.setSingleStep(10)
        wavelength_layout.addWidget(self.wavelength_min)
        wavelength_layout.addWidget(QLabel("to"))
        self.wavelength_max = QDoubleSpinBox()
        self.wavelength_max.setRange(200, 2000)
        self.wavelength_max.setValue(700)
        self.wavelength_max.setSingleStep(10)
        wavelength_layout.addWidget(self.wavelength_max)
        param_layout.addLayout(wavelength_layout)

        # Pixel size range
        pixel_layout = QHBoxLayout()
        pixel_layout.addWidget(QLabel("Pixel Size Range (μm):"))
        self.pixel_min = QDoubleSpinBox()
        self.pixel_min.setRange(0.1, 10.0)
        self.pixel_min.setValue(1.0)
        self.pixel_min.setSingleStep(0.1)
        pixel_layout.addWidget(self.pixel_min)
        pixel_layout.addWidget(QLabel("to"))
        self.pixel_max = QDoubleSpinBox()
        self.pixel_max.setRange(0.1, 10.0)
        self.pixel_max.setValue(5.0)
        self.pixel_max.setSingleStep(0.1)
        pixel_layout.addWidget(self.pixel_max)
        param_layout.addLayout(pixel_layout)

        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        # Output Directory Section
        output_group = QGroupBox("2. Output Directory")
        output_layout = QVBoxLayout()
        self.output_dir_btn = QPushButton("Select Output Directory")
        self.output_dir_btn.clicked.connect(self.select_output_directory)
        output_layout.addWidget(self.output_dir_btn)
        self.output_dir_label = QLabel("No directory selected")
        output_layout.addWidget(self.output_dir_label)
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # Generate Dataset Button
        self.generate_dataset_btn = QPushButton("Generate Dataset")
        self.generate_dataset_btn.clicked.connect(self.generate_dataset)
        layout.addWidget(self.generate_dataset_btn)

        tab.setLayout(layout)
        return tab

    def select_output_directory(self):
        """Select the output directory for saving the dataset."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_dir_label.setText(f"Selected: {dir_path}")
            self.output_dir = dir_path

    def generate_dataset(self):
        """Generate a dataset by sweeping parameters."""
        try:
            if not hasattr(self, 'output_dir'):
                QMessageBox.warning(self, "Warning", "Please select an output directory first.")
                return

            # Get parameter ranges
            wavelength_range = np.linspace(self.wavelength_min.value(), self.wavelength_max.value(), 10)
            pixel_size_range = np.linspace(self.pixel_min.value(), self.pixel_max.value(), 10)

            # Generate dataset
            for wavelength in wavelength_range:
                for pixel_size in pixel_size_range:
                    # Generate phase mask
                    phase_mask = self.phase_generator.generate(
                        wavelength=wavelength * 1e-9,  # Convert to meters
                        pixel_size=pixel_size * 1e-6  # Convert to meters
                    )
                    # Save to file
                    file_name = f"mask_{int(wavelength)}nm_{pixel_size:.2f}um.npy"
                    file_path = os.path.join(self.output_dir, file_name)
                    np.save(file_path, phase_mask)

            QMessageBox.information(self, "Success", "Dataset generated successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate dataset: {str(e)}")

    def select_dataset_file(self):
        """Select the dataset file for training."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Dataset File", "", "Numpy Files (*.npy);;All Files (*)"
        )
        if file_path:
            self.dataset_path_label.setText(f"Selected: {file_path}")
            self.dataset_path = file_path

    def fourier_optics_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        # Workspace engine integration
        self.workspace_scene = WorkspaceScene()
        self.workspace_view = WorkspaceView(self.workspace_scene)
        layout.addWidget(self.workspace_view, stretch=1)

        # Add component button (for demo)
        add_lens_btn = QPushButton("Add Lens")
        add_lens_btn.clicked.connect(self.add_lens_to_workspace)
        layout.addWidget(add_lens_btn)

        tab.setLayout(layout)
        return tab

    def add_lens_to_workspace(self):
        lens = LensItem()
        self.workspace_scene.addItem(lens)

    def execute_workflow(self):
        """Execute a workflow by calling the backend API."""
        try:
            # Define the workflow in JSON format
            workflow_json = """
            {
                "nodes": [
                    {
                        "id": "simulate_fdtd",
                        "type": "simulate",
                        "params": {
                            "simulation_type": "FDTD",
                            "wavelength": 500e-9,
                            "grid_size": 100
                        }
                    },
                    {
                        "id": "save_data",
                        "type": "save_data",
                        "params": {
                            "input": "simulate_fdtd",
                            "filename": "fdtd_result.npy"
                        }
                    }
                ]
            }
            """

            # Call the backend API
            url = f"{self.backend_url}/execute_workflow"
            headers = {'Content-Type': 'application/json'}
            data = {'workflow_json': workflow_json}
            response = requests.post(url, headers=headers, data=json.dumps(data))

            # Process the response
            if response.status_code == 200:
                results = response.json()['results']
                # Display the results in the GUI
                self.generation_status.setText(f"Workflow Results: {results}")
            else:
                QMessageBox.critical(self, "Error", f"Backend Error: {response.text}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to execute workflow: {str(e)}")

if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())




#app = FastAPI()

# Define data models
class SimulationType(str, Enum):
    FDTD = "FDTD"
    RAY_TRACING = "RAY_TRACING"

class SimulationRequest(BaseModel):
    simulation_type: SimulationType
    params: Dict

class OptimizationRequest(BaseModel):
    simulation_type: SimulationType
    wavelength_min: float
    wavelength_max: float
    grid_size_min: int
    grid_size_max: int
    n_trials: int

class WorkflowRequest(BaseModel):
    workflow_json: str

# Initialize modules
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
fdtd_simulator = FDTDSimulation(device=device)
ray_tracing_simulator = RayTracingSimulation()
data_manager = DataManager()
simulation_engine = {"FDTD": fdtd_simulator, "RAY_TRACING": ray_tracing_simulator}
workflow_automation = WorkflowAutomation(simulation_engine, data_manager)

# Simulation endpoint
@app.post("/simulate")
async def simulate(request: SimulationRequest):
    """
    Endpoint to run a simulation.
    """
    try:
        if request.simulation_type == SimulationType.FDTD:
            simulator = fdtd_simulator
        elif request.simulation_type == SimulationType.RAY_TRACING:
            simulator = ray_tracing_simulator
        else:
            raise ValueError("Invalid simulation type")

        result = simulator.simulate(request.params)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optimization endpoint
@app.post("/optimize")
async def optimize(params: OptimizationRequest):
    """
    Endpoint to run AutoML optimization.
    """
    try:
        def objective(trial):
            wavelength = trial.suggest_float("wavelength", params.wavelength_min, params.wavelength_max)
            grid_size = trial.suggest_int("grid_size", params.grid_size_min, params.grid_size_max)
            request_params = {"wavelength": wavelength, "grid_size": grid_size}

            if params.simulation_type == SimulationType.FDTD:
                simulator = fdtd_simulator
            elif params.simulation_type == SimulationType.RAY_TRACING:
                simulator = ray_tracing_simulator
            else:
                raise ValueError("Invalid simulation type")

            result = simulator.simulate(request_params)
            return result  # Replace with your actual optimization metric

        study = optuna.create_study(direction="minimize")  # Or "maximize"
        study.optimize(objective, n_trials=params.n_trials)

        best_params = study.best_params
        best_value = study.best_value
        return {"best_params": best_params, "best_value": best_value}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Workflow endpoint
@app.post("/execute_workflow")
async def execute_workflow(request: WorkflowRequest):
    """
    Endpoint to execute a workflow.
    """
    try:
        results = workflow_automation.execute_workflow(request.workflow_json)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "OK"}


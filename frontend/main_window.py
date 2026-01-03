"""
FourierLab - Advanced Signal Processing and Analysis Tool
A modern PyQt5 application showcasing technical excellence
"""

import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QStatusBar, QPushButton, 
                            QTextEdit, QFrame, QGridLayout, QProgressBar,
                            QTabWidget, QGroupBox)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon
import torch
import platform
import psutil
from datetime import datetime
from foaml.training.trainer import OpticsTrainer
from foaml.models.my_model import MyModel  # Example import
from PyQt6 import QtWidgets
from core.ai_assistant.assistant import OptiqAssistant
from core.optics_engine.transformations import MatrixTransformations
from ui.main_window import OptiqMainWindow


class SystemMonitorThread(QThread):
    """Background thread for system monitoring"""
    update_signal = pyqtSignal(dict)
    
    def run(self):
        while True:
            data = {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'gpu_temp': self.get_gpu_temp() if torch.cuda.is_available() else 0
            }
            self.update_signal.emit(data)
            self.msleep(1000)  # Update every second
    
    def get_gpu_temp(self):
        try:
            # This is a placeholder - actual GPU temp would require nvidia-ml-py
            return 45.0  # Mock temperature
        except:
            return 0.0

class ModernCard(QFrame):
    """Custom card widget with modern styling"""
    def __init__(self, title, content="", parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Box)
        self.setup_ui(title, content)
        
    def setup_ui(self, title, content):
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel(title)
        title_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Content
        if content:
            content_label = QLabel(content)
            content_label.setFont(QFont("Segoe UI", 10))
            content_label.setAlignment(Qt.AlignCenter)
            content_label.setWordWrap(True)
            layout.addWidget(content_label)

class FourierLabMainWindow(QMainWindow):
    """Modern main window with professional styling and features"""
    
    def __init__(self):
        super().__init__()
        self.system_monitor_thread = None
        self.init_ui()
        self.apply_dark_theme()
        self.start_system_monitoring()
        
        # Initialize the model and trainer
        model = MyModel()  # Replace with your actual model class
        self.trainer = OpticsTrainer(model, wavelength=632.8e-9, pixel_size=5e-6)
        
        # Dummy data for demonstration
        self.train_data = torch.randn(100, 1, 256, 256)
        self.train_targets = torch.randn(100, 2, 256, 256)
        self.val_data = torch.randn(20, 1, 256, 256)
        self.val_targets = torch.randn(20, 2, 256, 256)
        self.config = {
            "epochs": 5,
            "patience": 2,
            "batch_size": 8,
            "auto_tune": False
        }
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("FourierLab - Signal Processing Suite")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(1200, 800)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Left panel - System Info & Controls
        self.create_left_panel(main_layout)
        
        # Right panel - Main content area
        self.create_right_panel(main_layout)
        
        # Status bar
        self.create_status_bar()
        
    def create_left_panel(self, parent_layout):
        """Create the left information panel"""
        left_panel = QWidget()
        left_panel.setFixedWidth(350)
        left_layout = QVBoxLayout(left_panel)
        
        # System Information Card
        sys_info_card = self.create_system_info_card()
        left_layout.addWidget(sys_info_card)
        
        # CUDA Information Card
        cuda_card = self.create_cuda_info_card()
        left_layout.addWidget(cuda_card)
        
        # Performance Monitoring Card
        perf_card = self.create_performance_card()
        left_layout.addWidget(perf_card)
        
        # Control Panel
        control_card = self.create_control_panel()
        left_layout.addWidget(control_card)
        
        left_layout.addStretch()
        parent_layout.addWidget(left_panel)
        
    def create_system_info_card(self):
        """Create system information display card"""
        group = QGroupBox("System Information")
        layout = QVBoxLayout(group)
        
        # System details
        system_info = f"""
        <b>Platform:</b> {platform.system()} {platform.release()}<br>
        <b>Processor:</b> {platform.processor()}<br>
        <b>Architecture:</b> {platform.architecture()[0]}<br>
        <b>Python:</b> {platform.python_version()}<br>
        <b>PyTorch:</b> {torch.__version__}
        """
        
        info_label = QLabel(system_info)
        info_label.setWordWrap(True)
        info_label.setFont(QFont("Consolas", 9))
        layout.addWidget(info_label)
        
        return group
        
    def create_cuda_info_card(self):
        """Create CUDA information display card"""
        group = QGroupBox("GPU Acceleration")
        layout = QVBoxLayout(group)
        
        self.cuda_status_label = QLabel()
        self.cuda_status_label.setFont(QFont("Consolas", 9))
        self.update_cuda_status()
        layout.addWidget(self.cuda_status_label)
        
        return group
        
    def create_performance_card(self):
        """Create performance monitoring card"""
        group = QGroupBox("Performance Monitor")
        layout = QVBoxLayout(group)
        
        # CPU Usage
        cpu_layout = QHBoxLayout()
        cpu_layout.addWidget(QLabel("CPU:"))
        self.cpu_progress = QProgressBar()
        self.cpu_progress.setMaximum(100)
        cpu_layout.addWidget(self.cpu_progress)
        layout.addLayout(cpu_layout)
        
        # Memory Usage
        mem_layout = QHBoxLayout()
        mem_layout.addWidget(QLabel("RAM:"))
        self.memory_progress = QProgressBar()
        self.memory_progress.setMaximum(100)
        mem_layout.addWidget(self.memory_progress)
        layout.addLayout(mem_layout)
        
        # GPU Temperature (if available)
        if torch.cuda.is_available():
            gpu_layout = QHBoxLayout()
            gpu_layout.addWidget(QLabel("GPU:"))
            self.gpu_temp_label = QLabel("--°C")
            gpu_layout.addWidget(self.gpu_temp_label)
            layout.addLayout(gpu_layout)
        
        return group
        
    def create_control_panel(self):
        """Create control panel with action buttons"""
        group = QGroupBox("Control Panel")
        layout = QVBoxLayout(group)
        
        # Buttons
        buttons_config = [
            ("🚀 Initialize Engine", self.initialize_engine),
            ("📊 Run Analysis", self.run_analysis),
            ("🔧 Configure Settings", self.configure_settings),
            ("📈 View Results", self.view_results)
        ]
        
        for text, callback in buttons_config:
            button = QPushButton(text)
            button.clicked.connect(callback)
            button.setMinimumHeight(35)
            layout.addWidget(button)
            
        return group
        
    def create_right_panel(self, parent_layout):
        """Create the main content area"""
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Welcome header
        welcome_label = QLabel("Welcome to FourierLab")
        welcome_label.setFont(QFont("Segoe UI", 24, QFont.Bold))
        welcome_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(welcome_label)
        
        # Subtitle
        subtitle = QLabel("Advanced Signal Processing & Analysis Suite")
        subtitle.setFont(QFont("Segoe UI", 12))
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #888888; margin-bottom: 30px;")
        right_layout.addWidget(subtitle)
        
        # Tab widget for different sections
        self.tab_widget = QTabWidget()
        self.create_tabs()
        right_layout.addWidget(self.tab_widget)
        
        parent_layout.addWidget(right_panel, 1)  # Give right panel more space
        
    def create_tabs(self):
        """Create tabbed interface"""
        # Dashboard Tab
        dashboard_tab = QWidget()
        self.tab_widget.addTab(dashboard_tab, "📊 Dashboard")
        self.setup_dashboard_tab(dashboard_tab)
        
        # Analysis Tab
        analysis_tab = QWidget()
        self.tab_widget.addTab(analysis_tab, "🔬 Analysis")
        self.setup_analysis_tab(analysis_tab)
        
        # Results Tab
        results_tab = QWidget()
        self.tab_widget.addTab(results_tab, "📈 Results")
        self.setup_results_tab(results_tab)
        
        # Settings Tab
        settings_tab = QWidget()
        self.tab_widget.addTab(settings_tab, "⚙️ Settings")
        self.setup_settings_tab(settings_tab)
        
    def setup_dashboard_tab(self, tab):
        """Setup dashboard tab content"""
        layout = QVBoxLayout(tab)
        
        # Feature cards
        cards_layout = QGridLayout()
        
        features = [
            ("Real-time Processing", "Process signals in real-time with GPU acceleration"),
            ("Fourier Analysis", "Advanced FFT algorithms with customizable parameters"),
            ("Machine Learning", "AI-powered signal classification and prediction"),
            ("Visualization", "Interactive plots and spectrograms")
        ]
        
        for i, (title, desc) in enumerate(features):
            card = ModernCard(title, desc)
            cards_layout.addWidget(card, i // 2, i % 2)
            
        layout.addLayout(cards_layout)
        layout.addStretch()
        
    def setup_analysis_tab(self, tab):
        """Setup analysis tab content"""
        layout = QVBoxLayout(tab)
        
        info_label = QLabel("Analysis Tools")
        info_label.setFont(QFont("Segoe UI", 16, QFont.Bold))
        layout.addWidget(info_label)
        
        self.analysis_output = QTextEdit()
        self.analysis_output.setPlaceholderText("Analysis results will appear here...")
        self.analysis_output.setFont(QFont("Consolas", 10))
        layout.addWidget(self.analysis_output)
        
    def setup_results_tab(self, tab):
        """Setup results tab content"""
        layout = QVBoxLayout(tab)
        
        results_label = QLabel("Results & Visualization")
        results_label.setFont(QFont("Segoe UI", 16, QFont.Bold))
        layout.addWidget(results_label)
        
        placeholder = QLabel("📈 Results visualization will be displayed here")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet("color: #666666; font-size: 14px; padding: 50px;")
        layout.addWidget(placeholder)
        
    def setup_settings_tab(self, tab):
        """Setup settings tab content"""
        layout = QVBoxLayout(tab)
        
        settings_label = QLabel("Configuration Settings")
        settings_label.setFont(QFont("Segoe UI", 16, QFont.Bold))
        layout.addWidget(settings_label)
        
        settings_info = QLabel("⚙️ Application settings and preferences")
        settings_info.setAlignment(Qt.AlignCenter)
        settings_info.setStyleSheet("color: #666666; font-size: 14px; padding: 50px;")
        layout.addWidget(settings_info)
        
    def create_status_bar(self):
        """Create and configure status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Add permanent widgets to status bar
        self.time_label = QLabel()
        self.update_time()
        self.status_bar.addPermanentWidget(self.time_label)
        
        # Timer for time updates
        self.time_timer = QTimer()
        self.time_timer.timeout.connect(self.update_time)
        self.time_timer.start(1000)
        
        self.status_bar.showMessage("FourierLab initialized successfully")
        
    def start_system_monitoring(self):
        """Start background system monitoring"""
        self.system_monitor_thread = SystemMonitorThread()
        self.system_monitor_thread.update_signal.connect(self.update_performance_display)
        self.system_monitor_thread.start()
        
    def update_performance_display(self, data):
        """Update performance indicators"""
        self.cpu_progress.setValue(int(data['cpu_percent']))
        self.memory_progress.setValue(int(data['memory_percent']))
        
        if hasattr(self, 'gpu_temp_label') and torch.cuda.is_available():
            self.gpu_temp_label.setText(f"{data['gpu_temp']:.1f}°C")
            
    def update_cuda_status(self):
        """Update CUDA status display"""
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            status_text = f"""
            <div style='color: #4CAF50;'>
            <b>✅ CUDA Available</b><br>
            <b>Device:</b> {device_name}<br>
            <b>CUDA Version:</b> {cuda_version}<br>
            <b>Memory:</b> {memory_total:.1f} GB<br>
            <b>Status:</b> Ready for acceleration
            </div>
            """
        else:
            status_text = """
            <div style='color: #FF6B6B;'>
            <b>❌ CUDA Not Available</b><br>
            <b>Status:</b> CPU mode only<br>
            <b>Performance:</b> Limited
            </div>
            """
        
        self.cuda_status_label.setText(status_text)
        
    def update_time(self):
        """Update time display in status bar"""
        current_time = datetime.now().strftime("%H:%M:%S")
        self.time_label.setText(f"🕒 {current_time}")
        
    # Button callback methods
    def initialize_engine(self):
        self.status_bar.showMessage("Initializing processing engine...")
        self.analysis_output.append("🚀 Engine initialization started...")
        
    def run_analysis(self):
        self.status_bar.showMessage("Running ML analysis...")
        self.analysis_output.append("📊 ML analysis started...")

        train_loader = self.trainer.create_dataloader(self.train_data, self.train_targets, batch_size=self.config["batch_size"])
        val_loader = self.trainer.create_dataloader(self.val_data, self.val_targets, batch_size=self.config["batch_size"], shuffle=False)

        results = self.trainer.manual_train(train_loader, val_loader, self.config)
        self.analysis_output.append(f"Results: {results}")
        
    def configure_settings(self):
        self.status_bar.showMessage("Opening configuration panel...")
        self.tab_widget.setCurrentIndex(3)  # Switch to settings tab
        
    def view_results(self):
        self.status_bar.showMessage("Loading results viewer...")
        self.tab_widget.setCurrentIndex(2)  # Switch to results tab
        
    def apply_dark_theme(self):
        """Apply modern dark theme styling"""
        self.setStyleSheet("""
        QMainWindow {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        
        QWidget {
            background-color: #1e1e1e;
            color: #ffffff;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        
        QGroupBox {
            font-weight: bold;
            border: 2px solid #3d3d3d;
            border-radius: 8px;
            margin-top: 1ex;
            padding-top: 10px;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
            color: #ffffff;
        }
        
        QPushButton {
            background-color: #0d7377;
            border: none;
            border-radius: 6px;
            padding: 8px 16px;
            color: white;
            font-weight: bold;
            font-size: 10pt;
        }
        
        QPushButton:hover {
            background-color: #14a085;
        }
        
        QPushButton:pressed {
            background-color: #0a5d61;
        }
        
        QTabWidget::pane {
            border: 1px solid #3d3d3d;
            border-radius: 6px;
            background-color: #2d2d2d;
        }
        
        QTabBar::tab {
            background-color: #3d3d3d;
            border: 1px solid #555555;
            border-bottom-color: #3d3d3d;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            min-width: 120px;
            padding: 8px 12px;
            margin-right: 2px;
        }
        
        QTabBar::tab:selected {
            background-color: #0d7377;
            border-color: #0d7377;
        }
        
        QTabBar::tab:hover:!selected {
            background-color: #4d4d4d;
        }
        
        QProgressBar {
            border: 2px solid #3d3d3d;
            border-radius: 5px;
            text-align: center;
            background-color: #2d2d2d;
        }
        
        QProgressBar::chunk {
            background-color: #0d7377;
            border-radius: 3px;
        }
        
        QTextEdit {
            background-color: #2d2d2d;
            border: 1px solid #3d3d3d;
            border-radius: 6px;
            padding: 8px;
            font-family: 'Consolas', monospace;
        }
        
        QLabel {
            color: #ffffff;
        }
        
        QStatusBar {
            background-color: #2d2d2d;
            border-top: 1px solid #3d3d3d;
            color: #ffffff;
        }
        
        QFrame[frameShape="4"] {
            background-color: #2d2d2d;
            border: 1px solid #3d3d3d;
            border-radius: 8px;
            padding: 15px;
            margin: 5px;
        }
        """)
        
    def closeEvent(self, event):
        """Handle application close event"""
        if self.system_monitor_thread:
            self.system_monitor_thread.terminate()
            self.system_monitor_thread.wait()
        event.accept()
    
    def main():
    app = QtWidgets.QApplication(sys.argv)
    assistant = OptiqAssistant()
    optics = MatrixTransformations()
    window = OptiqMainWindow(assistant, optics)
    window.show()
    sys.exit(app.exec())


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("FourierLab")
    app.setOrganizationName("TechExcellence")
    
    # Set application icon (you can add an icon file later)
    # app.setWindowIcon(QIcon("icon.png"))
    
    window = FourierLabMainWindow()
    window.show()
    
    sys.exit(app.exec_())

self.ai_mentor = AIMentorWidget()
self.addDockWidget(Qt.RightDockWidgetArea, self.ai_mentor)

# gui/main_window.py
from PyQt6.QtWidgets import (QMainWindow, QApplication, QWidget, QPushButton, QVBoxLayout, QTextEdit,
                             QLabel, QHBoxLayout, QListWidget, QTabWidget, QLineEdit, QMessageBox)
import sys, os, json
from backend.optics_engine.optics_simulation import OpticsSimulation
from backend.optics_engine.matrix_transformer import MatrixTransformer
from backend.optics_engine.matrix_explainer import MatrixExplainer
from backend.ai_engine.assistant import OptiqAIAssistant
from backend.decision_framework.problem_tree import ProblemNode
from backend.decision_framework.decision_matrix import DecisionMatrix
from backend.decision_framework.summary_generator import generate_markdown_summary
from backend.experiment_logger import make_run_dir, save_run_meta
from utils.logger import setup_logger

logger = setup_logger()

class OptiqAIMain(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OptiqAI v0.1")
        self.resize(1000, 700)

        # backend objects
        self.sim = OpticsSimulation("Gaussian Demo")
        self.transformer = MatrixTransformer()
        self.explainer = MatrixExplainer()
        self.assistant = OptiqAIAssistant()
        
        # UI layout
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Tab: Simulation
        self.tab_sim = QWidget(); sim_layout = QVBoxLayout()
        self.run_btn = QPushButton("Run Gaussian Simulation")
        self.run_btn.clicked.connect(self.run_sim)
        self.vis_btn = QPushButton("Visualize Last")
        self.vis_btn.clicked.connect(self.sim.visualize)
        sim_layout.addWidget(self.run_btn); sim_layout.addWidget(self.vis_btn)
        self.sim_log = QTextEdit(); self.sim_log.setReadOnly(True)
        sim_layout.addWidget(self.sim_log)
        self.tab_sim.setLayout(sim_layout)
        self.tabs.addTab(self.tab_sim, "Simulation")

        # Tab: AI Assistant
        self.tab_ai = QWidget(); ai_layout = QVBoxLayout()
        self.ai_input = QLineEdit(); self.ai_input.setPlaceholderText("Type a question (e.g., 'ABCD matrix')")
        self.ask_btn = QPushButton("Ask Assistant")
        self.ask_btn.clicked.connect(self.ask_assistant)
        self.teach_input = QTextEdit(); self.teach_input.setPlaceholderText("Teach assistant (paste notes or equation)...")
        self.teach_btn = QPushButton("Teach Assistant")
        self.teach_btn.clicked.connect(self.teach_assistant)
        ai_layout.addWidget(self.ai_input)
        ai_layout.addWidget(self.ask_btn)
        ai_layout.addWidget(QLabel("Teach / Add to memory:"))
        ai_layout.addWidget(self.teach_input)
        ai_layout.addWidget(self.teach_btn)
        self.ai_out = QTextEdit(); self.ai_out.setReadOnly(True)
        ai_layout.addWidget(self.ai_out)
        self.tab_ai.setLayout(ai_layout)
        self.tabs.addTab(self.tab_ai, "AI Assistant")

        # Tab: Decision
        self.tab_dec = QWidget(); dec_layout = QVBoxLayout()
        self.dec_list = QListWidget()
        self.new_dec_btn = QPushButton("Run Example Decision: choose denoising method")
        self.new_dec_btn.clicked.connect(self.run_decision_example)
        dec_layout.addWidget(self.new_dec_btn)
        dec_layout.addWidget(self.dec_list)
        self.tab_dec.setLayout(dec_layout)
        self.tabs.addTab(self.tab_dec, "Decision Framework")

        # Tab: Matrix Companion
        self.tab_mat = QWidget(); mat_layout = QVBoxLayout()
        self.mat_input = QLineEdit(); self.mat_input.setPlaceholderText("Enter matrix key to explain (ABCD, Jones, Mueller, Fourier)")
        self.mat_btn = QPushButton("Explain Matrix")
        self.mat_btn.clicked.connect(self.explain_matrix)
        self.mat_out = QTextEdit(); self.mat_out.setReadOnly(True)
        self.mat_of_day = QPushButton("Matrix of the Day")
        self.mat_of_day.clicked.connect(self.show_matrix_of_day)
        mat_layout.addWidget(self.mat_input); mat_layout.addWidget(self.mat_btn); mat_layout.addWidget(self.mat_of_day); mat_layout.addWidget(self.mat_out)
        self.tab_mat.setLayout(mat_layout)
        self.tabs.addTab(self.tab_mat, "Matrix Companion")

        # Run list + persistence buttons
        bottom = QWidget(); bottom_layout = QHBoxLayout()
        self.save_run_btn = QPushButton("Save example run (log)")
        self.save_run_btn.clicked.connect(self.save_example_run)
        bottom_layout.addWidget(self.save_run_btn)
        bottom.setLayout(bottom_layout)
        sim_layout.addWidget(bottom)

    # Simulation handlers
    def run_sim(self):
        res = self.sim.run_gaussian_beam(waist=0.8)
        self.sim_log.append("Ran Gaussian simulation (waist=0.8). Results stored.")
        # brief summary
        self.sim_log.append(f"Result length: {len(res['x'])}")

    # AI assistant handlers
    def ask_assistant(self):
        q = self.ai_input.text().strip()
        if not q:
            QMessageBox.warning(self, "Empty query", "Please type a question.")
            return
        ans = self.assistant.ask(q)
        self.ai_out.append(f"Q: {q}\nA: {ans}\n---")

    def teach_assistant(self):
        txt = self.teach_input.toPlainText().strip()
        if not txt:
            QMessageBox.warning(self, "Empty", "Provide text to teach.")
            return
        self.assistant.teach(txt)
        self.ai_out.append("Added to assistant memory.\n")

    # Decision example
    def run_decision_example(self):
        options = ["Classical Tikhonov", "Small CNN", "PINN hybrid"]
        criteria = ["Accuracy", "Runtime", "Data required", "Explainability"]
        weights = [0.4, 0.2, 0.2, 0.2]
        dm = DecisionMatrix(options, criteria, weights)
        # fill scores (0-1) — example heuristic
        # Classical: accurate, fast, low data, high explainability
        dm.set_score(0,0,0.7); dm.set_score(0,1,0.9); dm.set_score(0,2,0.9); dm.set_score(0,3,0.9)
        # CNN: potentially accurate, medium runtime, high data, low explainability
        dm.set_score(1,0,0.8); dm.set_score(1,1,0.5); dm.set_score(1,2,0.2); dm.set_score(1,3,0.3)
        # PINN: accurate + physics, slower, medium data, medium explainability
        dm.set_score(2,0,0.85); dm.set_score(2,1,0.4); dm.set_score(2,2,0.5); dm.set_score(2,3,0.6)
        opt, score, all_scores = dm.recommend()
        self.dec_list.addItem(f"Recommendation: {opt} (score={score:.3f})")
        # save summary
        matrix_dict = {"options": options, "criteria": criteria, "weights": weights, "scores": all_scores.tolist()}
        rec = {"choice": opt, "score": score}
        path = generate_markdown_summary("Denoising Decision", matrix_dict, rec, outdir="runs")
        self.dec_list.addItem(f"Saved summary to {path}")

    # Matrix companion handlers
    def explain_matrix(self):
        key = self.mat_input.text().strip()
        if not key:
            QMessageBox.warning(self, "Enter key", "Enter ABCD, Jones, Mueller, or Fourier")
            return
        info = self.explainer.explain(key)
        self.mat_out.setText(json.dumps(info, indent=2))

    def show_matrix_of_day(self):
        import random
        keys = list(self.explainer.lib.keys())
        key = random.choice(keys)
        info = self.explainer.explain(key)
        self.mat_out.setText(f"Matrix of the Day: {key}\n\n" + json.dumps(info, indent=2))

    # persistence example
    def save_example_run(self):
        run_dir = make_run_dir(prefix="example")
        meta = {"name":"example run", "sim_meta": self.sim.meta}
        save_run_meta(run_dir, meta)
        QMessageBox.information(self, "Saved", f"Run saved to {run_dir}")

def run_gui():
    app = QApplication(sys.argv)
    win = OptiqAIMain()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
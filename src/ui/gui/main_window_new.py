from PyQt5.QtWidgets import (
    QMainWindow, QFileDialog, QVBoxLayout, QHBoxLayout, QWidget, 
    QLabel, QPushButton, QTabWidget, QSplitter, QSlider, QComboBox,
    QGroupBox, QGridLayout, QScrollArea, QFrame, QToolBar, QAction,
    QDockWidget, QStatusBar, QMenu, QMenuBar
)
from PyQt5.QtCore import Qt, QSize, pyqtSignal, QThread
from PyQt5.QtGui import QColor, QPalette, QFont, QIcon, QPixmap
import numpy as np
from torch import layout

class FourierLabGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FourierLab - Advanced Optics Analysis")
        self.setMinimumSize(1200, 800)
        
        # Define color palette based on the design spec
        self.colors = {
            'background': QColor("#0F172A"),  # Dark mode base
            'electric_field': QColor("#4F46E5"),  # Blue-purple for E-field
            'intensity': QColor("#F59E0B"),  # Amber for intensity
            'phase': QColor("#2DD4BF"),  # Teal for phase
            'text': QColor("#E2E8F0"),  # Light text for dark mode
            'panel': QColor("#1E293B")  # Slightly lighter than background
        }
        
        # Set the dark mode palette
        self.set_dark_mode()
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create toolbar
        self.create_toolbar()
        
        # Create status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready for simulation")
        
        # Create the main layout
        self.setup_ui()

    def set_dark_mode(self):
        """Apply dark mode styling to the entire application"""
        palette = QPalette()
        palette.setColor(QPalette.Window, self.colors['background'])
        palette.setColor(QPalette.WindowText, self.colors['text'])
        palette.setColor(QPalette.Base, self.colors['panel'])
        palette.setColor(QPalette.AlternateBase, self.colors['background'])
        palette.setColor(QPalette.ToolTipBase, self.colors['panel'])
        palette.setColor(QPalette.ToolTipText, self.colors['text'])
        palette.setColor(QPalette.Text, self.colors['text'])
        palette.setColor(QPalette.Button, self.colors['panel'])
        palette.setColor(QPalette.ButtonText, self.colors['text'])
        palette.setColor(QPalette.Link, self.colors['electric_field'])
        palette.setColor(QPalette.Highlight, self.colors['electric_field'])
        palette.setColor(QPalette.HighlightedText, self.colors['text'])
        
        # Apply the palette
        self.setPalette(palette)
        
        # Set style sheet for more detailed styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0F172A;
            }
            QPushButton {
                background-color: #334155;
                color: #E2E8F0;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4F46E5;
            }
            QPushButton:pressed {
                background-color: #3730A3;
            }
            QLabel {
                color: #E2E8F0;
                font-size: 12px;
            }
            QGroupBox {
                border: 1px solid #334155;
                border-radius: 4px;
                margin-top: 12px;
                font-weight: bold;
                color: #E2E8F0;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 5px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #334155;
                height: 8px;
                background: #1E293B;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4F46E5;
                border: 1px solid #4F46E5;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
            QTabWidget::pane {
                border: 1px solid #334155;
                border-radius: 4px;
                top: -1px;
            }
            QTabBar::tab {
                background: #1E293B;
                border: 1px solid #334155;
                padding: 8px 16px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                color: #E2E8F0;
            }
            QTabBar::tab:selected {
                background: #334155;
            }
            QComboBox {
                background-color: #1E293B;
                border: 1px solid #334155;
                border-radius: 4px;
                padding: 5px;
                min-width: 120px;
                color: #E2E8F0;
            }
            QComboBox::drop-down {
                border: 0px;
            }
            QComboBox QAbstractItemView {
                background-color: #1E293B;
                border: 1px solid #334155;
                selection-background-color: #4F46E5;
                color: #E2E8F0;
            }
            QScrollBar:vertical {
                background-color: #1E293B;
                width: 12px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background-color: #334155;
                min-height: 20px;
                border-radius: 6px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QDockWidget {
                titlebar-close-icon: url(close.png);
                titlebar-normal-icon: url(float.png);
            }
            QDockWidget::title {
                background: #1E293B;
                padding-left: 10px;
                padding-top: 4px;
                color: #E2E8F0;
            }
            QMenuBar {
                background-color: #1E293B;
                color: #E2E8F0;
            }
            QMenuBar::item {
                background-color: transparent;
            }
            QMenuBar::item:selected {
                background-color: #334155;
            }
            QMenu {
                background-color: #1E293B;
                color: #E2E8F0;
                border: 1px solid #334155;
            }
            QMenu::item:selected {
                background-color: #334155;
            }
            QToolBar {
                background-color: #1E293B;
                border: none;
                spacing: 3px;
            }
            QToolButton {
                background-color: transparent;
                border: none;
                padding: 5px;
                border-radius: 4px;
            }
            QToolButton:hover {
                background-color: #334155;
            }
            QToolButton:pressed {
                background-color: #4F46E5;
            }
            QStatusBar {
                background-color: #1E293B;
                color: #94A3B8;
            }
        """)

    def create_menu_bar(self):
        """Create the application menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        new_action = QAction('New Project', self)
        new_action.setShortcut('Ctrl+N')
        file_menu.addAction(new_action)
        
        open_action = QAction('Open...', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.handle_upload)
        file_menu.addAction(open_action)
        
        save_action = QAction('Save Project...', self)
        save_action.setShortcut('Ctrl+S')
        file_menu.addAction(save_action)
        
        export_action = QAction('Export Results...', self)
        export_action.setShortcut('Ctrl+E')
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu('Edit')
        
        # Add your original edit menu items here
        
        # View menu
        view_menu = menubar.addMenu('View')
        
        # Add your original view menu items here
        
        # Tools menu
        tools_menu = menubar.addMenu('Tools')
        
        # Add your original tools menu items here
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        docs_action = QAction('Documentation', self)
        docs_action.setShortcut('F1')
        help_menu.addAction(docs_action)
        
        about_action = QAction('About FourierLab', self)
        help_menu.addAction(about_action)

    def create_toolbar(self):
        """Create the main toolbar"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)
        
        # Add your original toolbar actions here, styled with the new theme
        
        # Example:
        new_action = QAction('New', self)
        new_action.setToolTip('Create New Project')
        toolbar.addAction(new_action)
        
        open_action = QAction('Open', self)
        open_action.setToolTip('Open Existing Project')
        toolbar.addAction(open_action)

    def setup_ui(self):
        """Set up the main UI components - REPLACE WITH YOUR ORIGINAL TAB STRUCTURE"""
        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Main layout
        main_layout = QVBoxLayout(main_widget)
        
        # Replace this with your original tab structure
        # For example, if you had tabs like "Simulation", "Data Analysis", "Training", etc.
        self.tabs = QTabWidget()
        
        # Add your original tabs here
        simulation_tab = self.create_simulation_tab()
        data_analysis_tab = self.create_data_analysis_tab()
        training_tab = self.create_training_tab()
        
        self.tabs.addTab(simulation_tab, "Simulation")
        self.tabs.addTab(data_analysis_tab, "Data Analysis")
        self.tabs.addTab(training_tab, "Training")
        
        main_layout.addWidget(self.tabs)

    # Replace these with your original tab creation methods
    def create_simulation_tab(self):
        """Create the simulation tab with your original functionality"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Add your original simulation tab components here
        # But style them with the new dark theme
        
        # Example:
        title = QLabel("Simulation Setup")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 15px;")
        layout.addWidget(title)
        
        # Original simulation controls go here
        
        return tab
        
    def create_data_analysis_tab(self):
        """Create the data analysis tab with your original functionality"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Add your original data analysis tab components here
        # But style them with the new dark theme
        
        return tab
        
    def create_training_tab(self):
        """Create the training tab with your original functionality"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Add your original training tab components here
        # But style them with the new dark theme
        
        # Example for a training setup:
        training_group = QGroupBox("Training Parameters")
        training_layout = QGridLayout(training_group)
        
        # Add your original training controls
        
        layout.addWidget(training_group)
        
        # Add original start training button
        train_btn = QPushButton("Start Training")
        train_btn.setStyleSheet("""
            background-color: #4F46E5;
            font-size: 14px;
            padding: 10px;
            min-height: 50px;
        """)
        layout.addWidget(train_btn)
        
        return tab

    def handle_upload(self):
        """Handle file upload dialog"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Files", 
            "", "All Files (*.*)"
        )
        if files:
            # Handle with your original file upload logic
            pass
    
        self.suggest_btn = QPushButton("Suggest Next Step")
        self.suggest_btn.clicked.connect(self.suggest_next)
        layout.addWidget(self.suggest_btn)

        
    
    def suggest_next(self):
        suggestion = self.assistant.suggest_next()
        self.chat_output.append(f"OptiqAI Suggestion: {suggestion}\n")


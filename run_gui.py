from PyQt5.QtWidgets import QApplication
import sys
from src.fourierlab.UI.gui.main_window import MainWindow  # Note: MainWindow instead of FourierLabGUI

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Use Fusion style for better looking widgets
    
    window = MainWindow()  # Use the MainWindow class from your previous implementation
    window.show()
    
    sys.exit(app.exec_())
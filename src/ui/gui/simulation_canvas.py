from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QLinearGradient

class SimulationCanvas(QWidget):
    """Custom widget for rendering optical simulations"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(600, 400)
        
        # Simulation parameters
        self.wavelength = 550  # nm
        self.intensity = 1.0
        self.phase = 0.0
        
        # Visualization settings
        self.view_mode = "electric_field"  # or "intensity", "phase"
        self.grid_visible = True
        self.colormap = "viridis"  # or "jet", "plasma", etc.
        
        # Animation timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)
        self.animation_active = False
        self.time_step = 0
        
        # Initialize with empty data
        self.data = None
        
    def start_animation(self):
        """Start the animation timer"""
        self.animation_active = True
        self.timer.start(50)  # 20 fps
        
    def stop_animation(self):
        """Stop the animation timer"""
        self.animation_active = False
        self.timer.stop()
        
    def update_simulation(self):
        """Update the simulation state for animation"""
        # This would calculate the next frame of the simulation
        self.time_step += 1
        self.phase = (self.time_step * 10) % 360  # Rotating phase for animation
        self.update()  # Trigger a repaint
        
    def set_wavelength(self, wavelength):
        """Set the wavelength for simulation"""
        self.wavelength = wavelength
        self.update()
        
    def set_view_mode(self, mode):
        """Set the visualization mode"""
        self.view_mode = mode
        self.update()
        
    def set_data(self, data):
        """Set the data to be visualized"""
        self.data = data
        self.update()
        
    def toggle_grid(self):
        """Toggle grid visibility"""
        self.grid_visible = not self.grid_visible
        self.update()
        
    def paintEvent(self, event):
        """Paint the simulation visualization"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Fill background
        painter.fillRect(self.rect(), QColor("#0F172A"))
        
        # Draw simulation visualization (placeholder)
        if self.data is None:
            self.draw_placeholder(painter)
        else:
            self.draw_simulation(painter)
            
        # Draw grid if enabled
        if self.grid_visible:
            self.draw_grid(painter)
            
        # Draw axes and labels
        self.draw_axes(painter)
        
    def draw_placeholder(self, painter):
        """Draw a placeholder when no data is available"""
        # Create a gradient background
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0, QColor("#0F172A"))
        gradient.setColorAt(1, QColor("#1E293B"))
        
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.NoPen)
        painter.drawRect(self.rect())
        
        # Draw placeholder text
        painter.setPen(QColor("#64748B"))
        painter.drawText(self.rect(), Qt.AlignCenter, "Simulation will appear here\nUse the parameters panel to configure")
        
        # Draw a circle representing a lens or aperture
        center_x = self.width() / 2
        center_y = self.height() / 2
        radius = min(self.width(), self.height()) / 4
        
        # Draw the aperture
        painter.setPen(QPen(QColor("#4F46E5"), 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(center_x - radius, center_y - radius, 
                           radius * 2, radius * 2)
        
        # Draw some light rays
        painter.setPen(QPen(QColor("#F59E0B"), 1, Qt.DashLine))
        for angle in range(0, 360, 30):
            rad = angle * 3.14159 / 180
            end_x = center_x + radius * 2.5 * np.cos(rad)
            end_y = center_y + radius * 2.5 * np.sin(rad)
            painter.drawLine(center_x, center_y, end_x, end_y)
        
    def draw_simulation(self, painter):
        """Draw the actual simulation data"""
        # This would render the actual simulation data using the chosen colormap
        # For now it's a placeholder
        
        # Example: create a wave pattern based on phase
        width = self.width()
        height = self.height()
        center_x = width / 2
        center_y = height / 2
        
        # Use different colors based on view mode
        if self.view_mode == "electric_field":
            color = QColor("#4F46E5")  # Blue-purple
        elif self.view_mode == "intensity":
            color = QColor("#F59E0B")  # Amber
        else:  # Phase
            color = QColor("#2DD4BF")  # Teal
            
        # Create a pattern
        painter.setPen(Qt.NoPen)
        
        for r in range(10, min(width, height) // 2, 20):
            # Vary opacity based on a wave pattern
            phase_factor = (r + self.phase) / 30.0
            intensity = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(phase_factor))
            
            color.setAlphaF(intensity)
            painter.setBrush(QBrush(color))
            
            painter.drawEllipse(center_x - r, center_y - r, r * 2, r * 2)
            
    def draw_grid(self, painter):
        """Draw a coordinate grid"""
        width = self.width()
        height = self.height()
        
        # Set up grid pen
        grid_pen = QPen(QColor("#334155"))
        grid_pen.setStyle(Qt.DotLine)
        painter.setPen(grid_pen)
        
        # Draw horizontal grid lines
        for y in range(0, height, 50):
            painter.drawLine(0, y, width, y)
            
        # Draw vertical grid lines
        for x in range(0, width, 50):
            painter.drawLine(x, 0, x, height)
            
    def draw_axes(self, painter):
        """Draw coordinate axes and labels"""
        width = self.width()
        height = self.height()
        
        # Set up axis pen
        axis_pen = QPen(QColor("#94A3B8"))
        painter.setPen(axis_pen)
        
        # Draw axes
        # X-axis
        painter.drawLine(0, height - 20, width, height - 20)
        # Y-axis
        painter.drawLine(20, 0, 20, height)
        
        # Draw labels
        painter.drawText(width - 70, height - 5, "Position (µm)")
        
        # Rotate text for y-axis label
        painter.save()
        painter.translate(5, height // 2)
        painter.rotate(-90)
        painter.drawText(0, 0, "Position (µm)")
        painter.restore()
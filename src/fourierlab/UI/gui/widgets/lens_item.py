from .base_component import OpticalComponentItem
from PyQt5.QtGui import QPainter, QColor, QBrush
from PyQt5.QtCore import QPointF

class LensItem(OpticalComponentItem):
    def __init__(self, position=QPointF(0, 0), focal_length=100.0):
        super().__init__("Lens", position, focal_length=focal_length)

    def paint(self, painter, option, widget):
        # Draw a simple lens shape
        painter.setBrush(QBrush(QColor(120, 180, 255, 200)))
        painter.drawEllipse(self.boundingRect())
        painter.setPen(QColor(0, 0, 120))
        painter.drawText(self.boundingRect(), 0, f"f={self.params.get('focal_length', 100.0)}") 
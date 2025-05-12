from PyQt5.QtWidgets import QGraphicsObject, QGraphicsItem, QStyleOptionGraphicsItem, QWidget, QMenu
from PyQt5.QtCore import QRectF, pyqtSignal, QPointF
from PyQt5.QtGui import QPainter, QBrush, QColor

class OpticalComponentItem(QGraphicsObject):
    parameters_changed = pyqtSignal(dict)

    def __init__(self, name, position=QPointF(0, 0), **params):
        super().__init__()
        self.name = name
        self.setPos(position)
        self.params = params
        self.setFlags(
            QGraphicsItem.ItemIsMovable |
            QGraphicsItem.ItemIsSelectable |
            QGraphicsItem.ItemSendsGeometryChanges
        )

    def boundingRect(self):
        return QRectF(-20, -20, 40, 40)  # Example size

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget):
        painter.setBrush(QBrush(QColor(100, 200, 255, 180)))
        painter.drawEllipse(self.boundingRect())
        painter.drawText(self.boundingRect(), 0, self.name)

    def contextMenuEvent(self, event):
        menu = QMenu()
        edit_action = menu.addAction("Edit Parameters")
        delete_action = menu.addAction("Delete")
        action = menu.exec_(event.screenPos())
        if action == edit_action:
            self.edit_parameters()
        elif action == delete_action:
            self.scene().removeItem(self)

    def edit_parameters(self):
        # Open a dialog to edit self.params
        pass  # To be implemented

    def mouseDoubleClickEvent(self, event):
        self.edit_parameters() 
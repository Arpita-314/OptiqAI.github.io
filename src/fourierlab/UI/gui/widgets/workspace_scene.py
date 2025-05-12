from PyQt5.QtWidgets import QGraphicsScene

class WorkspaceScene(QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSceneRect(-500, -500, 1000, 1000)  # Example size 
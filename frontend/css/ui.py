from PyQt6 import QtWidgets, QtCore
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QLineEdit, QPushButton, QTextEdit

class OptiqMainWindow(QtWidgets.QMainWindow):
    def __init__(self, assistant, optics_engine):
        super().__init__()
        self.assistant = assistant
        self.optics = optics_engine
        self.setWindowTitle("OptiqAI — Scientific Copilot")
        self.setGeometry(300, 200, 800, 600)
        self.initUI()

    def initUI(self):
        widget = QtWidgets.QWidget()
        layout = QVBoxLayout()

        self.query_label = QLabel("Ask OptiqAI:")
        self.input_box = QLineEdit()
        self.chat_output = QTextEdit()
        self.chat_output.setReadOnly(True)
        self.submit_btn = QPushButton("Send")
        self.submit_btn.clicked.connect(self.send_query)

        layout.addWidget(self.query_label)
        layout.addWidget(self.input_box)
        layout.addWidget(self.submit_btn)
        layout.addWidget(self.chat_output)

        widget.setLayout(layout)
        self.setCentralWidget(widget)

                tabs = QTabWidget()
        chat_tab = QWidget()
        chat_layout = QVBoxLayout()

        self.query_label = QLabel("Ask OptiqAI:")
        self.input_box = QLineEdit()
        self.chat_output = QTextEdit()
        self.chat_output.setReadOnly(True)
        self.submit_btn = QPushButton("Send")
        self.submit_btn.clicked.connect(self.send_query)

        self.suggest_btn = QPushButton("Suggest Next Step")
        self.suggest_btn.clicked.connect(self.suggest_next)

        chat_layout.addWidget(self.query_label)
        chat_layout.addWidget(self.input_box)
        chat_layout.addWidget(self.submit_btn)
        chat_layout.addWidget(self.suggest_btn)
        chat_layout.addWidget(self.chat_output)
        chat_tab.setLayout(chat_layout)

        self.education_tab = EducationPanel()

        tabs.addTab(chat_tab, "Chat")
        tabs.addTab(self.education_tab, "Educational Mode")

        self.setCentralWidget(tabs)

    def send_query(self):
        query = self.input_box.text()
        if query.strip() == "":
            return
        response = self.assistant.chat(query)
        self.chat_output.append(f"You: {query}")
        self.chat_output.append(f"OptiqAI: {response}\n")
        self.input_box.clear()

edu_text = """
<h2>💡 OptiqAI Educational Mode</h2>
<p>OptiqAI helps you understand when to use classical vs modern ML:</p>
<ul>
<li><b>Use classical ML</b> (SVM, regression) when data is small, structured, or physics-driven.</li>
<li><b>Avoid deep learning</b> (CNN, RNN, LSTM) unless you have large labeled datasets.</li>
<li><b>Physics-informed ML</b> ensures that predictions respect conservation laws and boundary conditions.</li>
<li>Matrix transformations model how optical rays or waves evolve — fundamental for lens design and tomography.</li>
</ul>
"""
class EducationPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        label = QLabel("Educational Insights")
        text = QTextEdit()
        text.setReadOnly(True)
        text.setHtml(edu_text)
        layout.addWidget(label)
        layout.addWidget(text)
        self.setLayout(layout)
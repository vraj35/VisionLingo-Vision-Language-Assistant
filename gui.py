import sys
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QPixmap
from utils import predict_image
from language_module import generate_description

class VisionLingo(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VisionLingo Assistant")
        self.layout = QVBoxLayout()

        self.label = QLabel("Upload an image to get a description")
        self.img_label = QLabel()
        self.result_label = QLabel()

        self.button = QPushButton("Upload Image")
        self.button.clicked.connect(self.load_image)

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.img_label)
        self.layout.addWidget(self.result_label)
        self.layout.addWidget(self.button)

        self.setLayout(self.layout)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName()
        if path:
            pixmap = QPixmap(path).scaled(300, 300)
            self.img_label.setPixmap(pixmap)
            label = predict_image(path)
            description = generate_description(label)
            self.result_label.setText(f"Class: {label}\n\nDescription: {description}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VisionLingo()
    window.show()
    sys.exit(app.exec_())

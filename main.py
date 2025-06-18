import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PIL import Image
from cnn_model import load_cnn_model, predict_image_class
from language_module import generate_description

class VisionLingoApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VisionLingo - Vision Language Assistant")
        self.setGeometry(100, 100, 400, 400)

        self.image_label = QLabel("Upload an image", self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(300, 300)

        self.predict_button = QPushButton("Upload Image")
        self.predict_button.clicked.connect(self.upload_image)

        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        self.result_label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.predict_button)
        layout.addWidget(self.result_label)

        self.setLayout(layout)

        # Load the pre-trained CNN model
        self.model = load_cnn_model()

    def upload_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.bmp)")
        if file_name:
            self.display_image(file_name)

            # Preprocess image and predict class
            img = Image.open(file_name).convert('L').resize((28, 28))
            img_array = np.array(img).reshape(1, 28, 28, 1) / 255.0
            predicted_class = predict_image_class(self.model, img_array)

            # Generate description from language module
            description = generate_description(predicted_class)

            # Show result
            self.result_label.setText(f"Prediction: {predicted_class}\n\nDescription: {description}")

    def display_image(self, file_path):
        image = QImage(file_path)
        pixmap = QPixmap.fromImage(image).scaled(300, 300, Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VisionLingoApp()
    window.show()
    sys.exit(app.exec_())

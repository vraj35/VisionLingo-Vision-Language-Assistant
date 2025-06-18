📸 VisionLingo – Vision-Language Assistant
VisionLingo is an AI-powered desktop application that combines image classification and natural language processing. It allows users to upload or draw images, predicts what the image represents using a CNN model, and then explains the prediction in simple language using a Naive Bayes-based text generation module. The interface is built using PyQt5 for a smooth user experience.

🚀 Features
🧠 CNN-based Image Classifier – Trained on MNIST to recognize handwritten digits.

🗣️ Naive Bayes Language Module – Generates simple textual descriptions for predictions.

🖼️ Interactive GUI – Built with PyQt5 for easy image upload and visualization.

🔁 Real-Time Prediction – Upload an image and get immediate visual and textual feedback.

🏗️ Project Structure
visionlingo/
│
├── main.py                 # GUI and application logic
├── cnn_model.py           # CNN loading and prediction functions
├── language_module.py     # Naive Bayes-based text generation
├── model/
│   └── digit_cnn.h5       # Trained CNN model (Keras format)
├── images/                # Sample images for testing
└── README.md              # Project documentation

📦 Requirements
pip install numpy tensorflow keras pillow PyQt5 scikit-learn

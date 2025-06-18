import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class_names = ['cat', 'dog', 'car', 'tree', 'book']  # match training folders

model = load_model('cnn_model.h5')

def predict_image(path):
    img = image.load_img(path, target_size=(64, 64))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    label = class_names[np.argmax(preds)]
    return label

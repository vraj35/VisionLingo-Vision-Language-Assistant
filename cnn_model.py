from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(5, activation='softmax')  # assume 5 classes
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train = datagen.flow_from_directory('dataset/train', target_size=(64, 64), batch_size=32,
                                        class_mode='categorical', subset='training')
    val = datagen.flow_from_directory('dataset/train', target_size=(64, 64), batch_size=32,
                                      class_mode='categorical', subset='validation')

    model = create_model()
    model.fit(train, validation_data=val, epochs=5)
    model.save('cnn_model.h5')

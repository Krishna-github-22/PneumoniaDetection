import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

# Define paths and constants
data_dir = 'D:/Pro/ML/Dataset/test'
normal_dir = os.path.join(data_dir, 'NORMAL')
pneumonia_dir = os.path.join(data_dir, 'PNEUMONIA')
img_size = (224, 224)
batch_size = 32

# Create an ImageDataGenerator for data augmentation and normalization
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Load and preprocess data using the ImageDataGenerator
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(train_generator[0][0], train_generator[0][1], test_size=0.2, random_state=42)

# Define the AlexNet model
def build_alexnet():
    model = models.Sequential()
    # Convolutional layers
    model.add(layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=(224, 224, 3)))
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2)))
    model.add(layers.Conv2D(256, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2)))
    model.add(layers.Conv2D(384, (3, 3), activation='relu'))
    model.add(layers.Conv2D(384, (3, 3), activation='relu'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2)))
    # Flatten and fully connected layers
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification

    return model

# Build and compile the model
alexnet_model = build_alexnet()
alexnet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
alexnet_model.fit(train_generator, epochs=3, validation_data=(X_test, y_test))

# Evaluate the model
predictions = (alexnet_model.predict(X_test) > 0.5).astype("int32")
print(classification_report(y_test, predictions))

_, accuracy = alexnet_model.evaluate(X_test, y_test)

print(f"Test Accuracy: {accuracy * 100:.2f}%")
history = alexnet_model.fit(train_generator, epochs=3, validation_data=(X_test, y_test))

# Evaluate the model


import tensorflow as tf
from tensorflow.keras import layers, models
# from sklearn.model_selection import traintest_split
from sklearn.metrics import classification_report
import os
import matplotlib.pyplot as plt
# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
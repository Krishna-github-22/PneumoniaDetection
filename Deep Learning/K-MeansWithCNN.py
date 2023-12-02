import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.preprocessing import label_binarize
# Define a custom Keras model transformer for compatibility with scikit-learn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
class KerasModelTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, build_fn, input_shape, epochs=5, batch_size=32):
        self.build_fn = build_fn
        self.input_shape = input_shape
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, X, y=None, **fit_params):
        self.model_ = self.build_fn(self.input_shape, self.epochs, self.batch_size)
        self.model_.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return self

    def transform(self, X):
        return self.model_.predict(X).reshape(-1, 1)  # Flatten the output

# Function to load and preprocess the chest X-ray dataset
def load_and_preprocess_data(data_dir):
    data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    data_flow = data_generator.flow_from_directory(
        data_dir,
        target_size=(224, 224),  # Adjust the target size based on your requirements
        batch_size=32,
        class_mode='binary',
        shuffle=False
    )
    images, labels = data_flow.next()
    return images, labels

# Build a simple CNN model
def build_cnn_model(input_shape, epochs=5, batch_size=32):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Specify hyperparameters for tuning
param_grid = {
    'cnn__epochs': [5, 10],
    'cnn__batch_size': [16, 32],
}

# Specify your dataset directory
data_directory = 'D:\Pro\ML\Dataset'

# Load and preprocess the dataset
x_train, y_train = load_and_preprocess_data(os.path.join(data_directory, 'train'))
x_test, y_test = load_and_preprocess_data(os.path.join(data_directory, 'test'))

# Create a pipeline with k-means clustering and CNN
pipeline = Pipeline([
    ('cnn', KerasModelTransformer(build_cnn_model, input_shape=(224, 224, 3))),
    ('kmeans', KMeans(n_clusters=2))
])

# Combine hyperparameter tuning with k-means clustering
grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=2, scoring='accuracy', verbose=2)
grid_result = grid_search.fit(x_train, y_train)

# Fit the model with the best hyperparameters on the entire dataset
best_model = grid_result.best_estimator_
best_model.fit(x_train, y_train)
# Evaluate the model
y_pred = best_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')


cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
disp.plot(cmap='viridis', values_format='.4g')
plt.title('Confusion Matrix')
plt.show()
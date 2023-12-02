import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Function to load and preprocess chest X-ray images
def load_and_preprocess_data(data_dir, img_size=(128, 128)):
    data = []
    labels = []

    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        for img_file in os.listdir(label_path):
            img_path = os.path.join(label_path, img_file)
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            img = img.resize(img_size)
            data.append(np.array(img).flatten())
            labels.append(label)

    data = np.array(data)
    labels = np.array(labels)

    return data, labels

# Load and preprocess data
data_dir = "D:/Pro/ML/Dataset/test"  # Update this path
data, labels = load_and_preprocess_data(data_dir)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Apply K-means clustering
num_clusters = 2  # Two classes: NORMAL and PNEUMONIA
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X_train)

# Assign cluster labels to the training and testing sets
X_train_clustered = kmeans.predict(X_train)
X_test_clustered = kmeans.predict(X_test)

# SVM classification
svm_classifier = svm.SVC()
svm_classifier.fit(X_train_clustered.reshape(-1, 1), y_train)

# Predict on the test set
y_pred = svm_classifier.predict(X_test_clustered.reshape(-1, 1))

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print results
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

import matplotlib.pyplot as plt

# Scatter plot for training data
plt.scatter(X_train_clustered, np.zeros_like(X_train_clustered), c='red', cmap='viridis', label='Training Data')
plt.scatter(X_test_clustered, np.ones_like(X_test_clustered), c='blue', cmap='viridis', marker='x', label='Test Predictions')
plt.title('SVM Classification on K-means Clusters')
plt.xlabel('Cluster Labels')
plt.ylabel('Data Points')
plt.legend()
plt.show()

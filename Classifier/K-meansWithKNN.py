import os
import numpy as np
from skimage import io, color, transform
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Function to read and preprocess images
def preprocess_images(image_folder, label):
    images = []
    labels = []
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpeg") or filename.endswith(".png"):
            img_path = os.path.join(image_folder, filename)
            img = io.imread(img_path, as_gray=True)  
            img = transform.resize(img, (100, 100))  
            images.append(img.flatten())  
            labels.append(label)
    return images, labels

normal_images, normal_labels = preprocess_images("D:/Pro/ML/Dataset/test/NORMAL/", 0)
pneumonia_images, pneumonia_labels = preprocess_images("D:/Pro/ML/Dataset/test/PNEUMONIA/", 1)

all_images = normal_images + pneumonia_images
all_labels = normal_labels + pneumonia_labels

X = np.array(all_images)
y = np.array(all_labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train)


X_train_clusters = kmeans.predict(X_train)
X_test_clusters = kmeans.predict(X_test)


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_clusters.reshape(-1, 1), y_train)


y_pred = knn.predict(X_test_clusters.reshape(-1, 1))

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")


import matplotlib.pyplot as plt

# ... (your existing code)

# Plot the clusters
plt.figure(figsize=(10, 6))

# Plot points in the training set
plt.scatter(X_train_clusters, y_train, c=kmeans.labels_, cmap='viridis', marker='o', label='Training Set')

# Plot points in the testing set
plt.scatter(X_test_clusters, y_test, c=kmeans.predict(X_test), cmap='viridis', marker='x', label='Testing Set')

plt.title('KMeans Clusters')
plt.xlabel('Cluster Assignments')
plt.ylabel('Class (0: NORMAL, 1: PNEUMONIA)')
plt.legend()
plt.show()

from sklearn.metrics import classification_report

# ... (your existing code)

class_report = classification_report(y_test, y_pred)
print(f"Classification Report:\n{class_report}")

# Extract sensitivity (recall) from the classification report
sensitivity = float(class_report.split('\n')[3].split()[3])
print(f"Sensitivity (Recall): {sensitivity}")

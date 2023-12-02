import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from skimage import io, color, transform


def load_images_from_folder(folder):
    images = []
    labels = []
    i=0
    for class_label in os.listdir(folder):
        class_path = os.path.join(folder, class_label)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                img_path = os.path.join(class_path, filename)
                img = io.imread(img_path)

                # Ensure the image has three channels
                if img.ndim == 2:
                    img = color.gray2rgb(img)

                img = transform.resize(img, (64, 64))  # Resize image for simplicity
                flat_img = img.flatten()  # Flatten the image into a 1D array
                images.append(flat_img)
                labels.append(class_label)
                print(i)
                i+=1
    return np.array(images), np.array(labels)

# Load images from the specified folder
folder_path = "D:/Pro/ML/Dataset/test"
X, y = load_images_from_folder(folder_path)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a Naïve Bayes classifier
nb_classifier = GaussianNB()
print("0")
# Train the classifier
nb_classifier.fit(X_train, y_train)
print("0")
# Make predictions on the test set
y_pred = nb_classifier.predict(X_test)
print("0")
# Evaluate the performance
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1_score = metrics.f1_score(y_test, y_pred, average='weighted')
print("0")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1_score:.2f}")


import matplotlib.pyplot as plt

# Assuming you have already calculated accuracy, precision, recall, and f1_score

# Create a bar graph
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
metrics_values = [accuracy, precision, recall, f1_score]

plt.bar(metrics_names, metrics_values, color=['blue', 'green', 'orange', 'red'])
plt.title('Model Performance Metrics')
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.ylim(0, 1)  # Assuming scores are between 0 and 1

# Display the values on top of the bars
for i, value in enumerate(metrics_values):
    plt.text(i, value + 0.01, f'{value:.2f}', ha='center', va='bottom')

plt.show()

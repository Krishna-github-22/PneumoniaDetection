import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

dataset_path = 'D:/Pro/ML/Dataset/train'

image_paths = []
labels = []

for label in os.listdir(dataset_path):
    label_path = os.path.join(dataset_path, label)
    if os.path.isdir(label_path):
        for image_file in os.listdir(label_path):
            image_paths.append(os.path.join(label_path, image_file))
            labels.append(label)

le = LabelEncoder()
labels = le.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42
)

def extract_features(image_path):
    return np.random.rand(10)

X_train_features = np.array([extract_features(image_path) for image_path in X_train])
X_test_features = np.array([extract_features(image_path) for image_path in X_test])

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train_features, y_train)

y_pred = clf.predict(X_test_features)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_rep)

plt.figure(figsize=(10, 10))
plot_tree(clf, filled=True, feature_names=[f"feature_{i}" for i in range(X_train_features.shape[1])], class_names=['NORMAL', 'PNEUMONIA'])
plt.show()

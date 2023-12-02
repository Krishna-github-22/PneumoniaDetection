import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import skfuzzy as fuzz
from PIL import Image
import matplotlib.pyplot as plt
print("0")
# Step 1: Load Chest X-ray images
def load_images(folder_path, target_size=(128, 128)):
    images = []
    labels = []
    class_mapping = {'NORMAL': 0, 'PNEUMONIA': 1}

    for class_label, subfolder in class_mapping.items():
        subfolder_path = os.path.join(folder_path, class_label)
        for filename in os.listdir(subfolder_path):
            img_path = os.path.join(subfolder_path, filename)
            img = Image.open(img_path).convert("L")  # Convert to grayscale
            img = img.resize(target_size)  # Resize the image to a fixed size
            img_array = np.array(img).flatten()
            images.append(img_array)
            labels.append(subfolder)
            print("0")
    return np.array(images), np.array(labels)
# Step 2: Apply Fuzzy C-Means clustering
def apply_fuzzy_cmeans(images, num_clusters, m=2, error=0.005, max_iter=100):
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        images.T,
        num_clusters,
        m,
        error=error,
        maxiter=max_iter,
        init=None
    )
    print("0")
    return cntr, u

# Step 3: Extract Fuzzy C-Means features
def extract_fuzzy_cmeans_features(images, cntr, u):
    features = fuzz.cluster.cmeans_predict(
        images.T,
        cntr,
        m=2,
        error=0.005,
        maxiter=100
    )[0].argmax(axis=0)
    print("0")
    return features

# Step 4: Split data for training and testing
def split_data(features, labels, test_size=0.2):
    return train_test_split(features, labels, test_size=test_size, random_state=42)

# Step 5: Train K-Nearest Neighbors (KNN) classifier
def train_knn_classifier(train_features, train_labels, n_neighbors=3):
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_classifier.fit(train_features, train_labels)
    return knn_classifier

# Step 6: Evaluate the model
def evaluate_model(classifier, test_features, test_labels):
    predictions = classifier.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    return accuracy

# Main program
if __name__ == "__main__":
    data_folder = "D:/Pro/ML/Dataset/test"
    num_clusters = 2  # Number of clusters for Fuzzy C-Means

    # Step 1: Load data
    images, labels = load_images(data_folder)

    # Step 2: Apply Fuzzy C-Means clustering
    cntr, u = apply_fuzzy_cmeans(images, num_clusters)
    print("0")
    # Step 3: Extract Fuzzy C-Means features
    # Step 3: Extract Fuzzy C-Means features
    fuzzy_cmeans_features = extract_fuzzy_cmeans_features(images, cntr, u)

    # Reshape the features to make them 2D
    fuzzy_cmeans_features = fuzzy_cmeans_features.reshape(-1, 1)

    # Step 4: Split data for training and testing
    train_features, test_features, train_labels, test_labels = split_data(fuzzy_cmeans_features, labels)

    # Step 5: Train KNN classifier
    knn_classifier = train_knn_classifier(train_features, train_labels)

    # Step 6: Evaluate the model
    accuracy = evaluate_model(knn_classifier, test_features, test_labels)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    plt.scatter(fuzzy_cmeans_features, labels, c=u.argmax(axis=0), cmap='viridis', s=50, alpha=0.5)
    plt.title('Fuzzy C-Means Clusters')
    plt.xlabel('Fuzzy C-Means Features')
    plt.ylabel('Class Labels')
    plt.show()
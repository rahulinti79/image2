import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os
import matplotlib.pyplot as plt

# Load dataset (assuming images are stored in a folder named 'images' with labels in a CSV file)
def load_dataset():
    images = []
    labels = []
    with open('C:/Users/rahul/Documents/Malignant_Melanoma', 'r') as file:
        for line in file:
            filename, label = line.strip().split(',')
            image = cv2.imread(os.path.join('images', filename))
            images.append(image)
            labels.append(int(label))
    return np.array(images), np.array(labels)

# Preprocess images (resize, normalize, etc.)
def preprocess_images(images):
    processed_images = []
    for image in images:
        processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed_image = cv2.resize(processed_image, (100, 100))
        processed_images.append(processed_image)
    return np.array(processed_images)

# Apply image processing techniques
def apply_image_processing(images):
    processed_images = []
    for image in images:
        processed_image = cv2.medianBlur(image, 5)
        processed_images.append(processed_image)
    return np.array(processed_images)

# Load and preprocess dataset
images, labels = load_dataset()
processed_images = preprocess_images(images)
processed_images = apply_image_processing(processed_images)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(processed_images, labels, test_size=0.2, random_state=42)

# Define and train the initial model (pure image processing-based approach)
model_initial = RandomForestClassifier(n_estimators=100, random_state=42)
model_initial.fit(X_train.reshape(X_train.shape[0], -1), y_train)

# Evaluate the initial model
y_pred_initial = model_initial.predict(X_test.reshape(X_test.shape[0], -1))
accuracy_initial = accuracy_score(y_test, y_pred_initial)
print(f'Accuracy (Initial): {accuracy_initial}')

# Define and train the machine learning-based model using OpenCV APIs (Bonus Credit)
# Example: Using the OpenCV SVM model
svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setType(cv2.ml.SVM_C_SVC)
svm.train(X_train.reshape(X_train.shape[0], -1), cv2.ml.ROW_SAMPLE, y_train)

# Evaluate the machine learning-based model
_, y_pred_opencv = svm.predict(X_test.reshape(X_test.shape[0], -1))
accuracy_opencv = accuracy_score(y_test, y_pred_opencv)
print(f'Accuracy (OpenCV ML): {accuracy_opencv}')

# Compare the two models
print(f'Difference in Accuracy: {accuracy_opencv - accuracy_initial}')

# Select a few random images from the test set
num_images_to_display = 5
random_indices = np.random.choice(range(len(X_test)), num_images_to_display, replace=False)
sample_images = X_test[random_indices]
sample_labels = y_test[random_indices]

# Predict labels for the sample images using both models
sample_predictions_initial = model_initial.predict(sample_images.reshape(num_images_to_display, -1))
_, sample_predictions_opencv = svm.predict(sample_images.reshape(num_images_to_display, -1))

# Display the sample images along with their predicted labels from both models
plt.figure(figsize=(15, 6))
for i in range(num_images_to_display):
    plt.subplot(2, num_images_to_display, i+1)
    plt.imshow(sample_images[i].reshape(100, 100), cmap='gray')
    plt.title(f'Initial: {sample_predictions_initial[i]}, Actual: {sample_labels[i]}')
    plt.axis('off')
    plt.subplot(2, num_images_to_display, num_images_to_display+i+1)
    plt.imshow(sample_images[i].reshape(100, 100), cmap='gray')
    plt.title(f'OpenCV ML: {sample_predictions_opencv[i]}, Actual: {sample_labels[i]}')
    plt.axis('off')
plt.show()

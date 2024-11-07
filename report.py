import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# Simulate a gait recognition dataset (replace with real data)
# Let's assume X contains sequences of shape (num_sequences, num_frames, height, width, channels)
# and y contains labels for each sequence
num_sequences = 500
num_frames = 30  # Number of frames in each sequence
height, width, channels = 64, 64, 1  # Dimensions of each frame
num_classes = 5  # Number of different people (classes)

# Randomly generate data (replace with actual gait data)
X = np.random.rand(num_sequences, num_frames, height, width, channels)
y = np.random.randint(0, num_classes, num_sequences)

# Normalize pixel values
X = X / 255.0

# One-hot encode the labels
label_binarizer = LabelBinarizer()
y = label_binarizer.fit_transform(y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the CRNN model
model = models.Sequential([
    # TimeDistributed CNN layers to extract features from each frame
    layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu'), input_shape=(num_frames, height, width, channels)),
    layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
    layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu')),
    layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
    layers.TimeDistributed(layers.Conv2D(128, (3, 3), activation='relu')),
    layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
    layers.TimeDistributed(layers.Flatten()),  # Flatten the CNN output

    # LSTM layers to process the sequence of features
    layers.LSTM(128, return_sequences=False),

    # Fully connected layer
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model and store the training history
history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.2f}")

# Predict on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Display classification report
print(classification_report(y_true_classes, y_pred_classes))

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper left')

# Show plots
plt.show()

# Show some sample test predictions
n_samples = 5
plt.figure(figsize=(10, 5))
for i in range(n_samples):
    plt.subplot(1, n_samples, i + 1)
    # Display the first frame of the sequence
    plt.imshow(X_test[i][0].reshape(height, width), cmap='gray')
    plt.title(f"Pred: {y_pred_classes[i]}\nTrue: {y_true_classes[i]}")
    plt.axis('off')

plt.show()
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Load the Flowers dataset from TensorFlow Datasets
(ds_train, ds_test), ds_info = tfds.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:]'],
    as_supervised=True,
    with_info=True
)

# Preprocess the dataset
def preprocess(image, label):
    image = tf.image.resize(image, (32, 32))  # Resize images to 32x32 pixels
    image = image / 255.0  # Normalize the pixel values
    label = tf.one_hot(label, ds_info.features['label'].num_classes)
    return image, label

# Apply preprocessing to the dataset
ds_train = ds_train.map(preprocess).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
ds_test = ds_test.map(preprocess).batch(32).prefetch(tf.data.experimental.AUTOTUNE)

# Create the MLP model
model = Sequential()
model.add(Flatten(input_shape=(32, 32, 3)))
model.add(Dense(128, activation='relu'))
model.add(Dense(ds_info.features['label'].num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model and capture the training history
history = model.fit(ds_train, epochs=10, validation_data=ds_test)

# Evaluate the model
loss, accuracy = model.evaluate(ds_test)
print('Test accuracy:', accuracy)

# Plot sample images with true labels
def plot_samples(dataset, num_samples=5):
    plt.figure(figsize=(10, 5))
    for i, (image, label) in enumerate(dataset.take(num_samples)):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(image.numpy())
        plt.title(f"Label: {tf.argmax(label)}")
        plt.axis('off')
    plt.show()

# Plot some sample images from the training dataset
plot_samples(ds_train.unbatch())

# Plot training & validation accuracy over epochs
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
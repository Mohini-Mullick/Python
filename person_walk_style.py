import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Set image dimensions
img_height, img_width = 224, 224
batch_size = 32

# Path to dataset
train_dir = 'path_to_train_data'  # Replace with your train dataset path
val_dir = 'path_to_validation_data'  # Replace with your validation dataset path

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

# Function to create and compile the model
def build_model(base_model):
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# VGG-16 model
vgg16_base = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
vgg16_model = build_model(vgg16_base)

# Train the VGG-16 model
vgg16_history = vgg16_model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# DenseNet-201 model
densenet_base = applications.DenseNet201(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
densenet_model = build_model(densenet_base)

# Train the DenseNet-201 model
densenet_history = densenet_model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# Function to plot training accuracy and loss
def plot_history(history, model_name):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# Plot the training history for VGG-16
plot_history(vgg16_history, 'VGG-16')

# Plot the training history for DenseNet-201
plot_history(densenet_history, 'DenseNet-201')

# Evaluate and show confusion matrix for the VGG-16 model
val_generator.reset()
vgg16_predictions = (vgg16_model.predict(val_generator) > 0.5).astype('int32')
vgg16_true_classes = val_generator.classes

print("VGG-16 Classification Report:")
print(classification_report(vgg16_true_classes, vgg16_predictions, target_names=val_generator.class_indices.keys()))

print("VGG-16 Confusion Matrix:")
print(confusion_matrix(vgg16_true_classes, vgg16_predictions))

# Evaluate and show confusion matrix for the DenseNet-201 model
val_generator.reset()
densenet_predictions = (densenet_model.predict(val_generator) > 0.5).astype('int32')
densenet_true_classes = val_generator.classes

print("DenseNet-201 Classification Report:")
print(classification_report(densenet_true_classes, densenet_predictions, target_names=val_generator.class_indices.keys()))

print("DenseNet-201 Confusion Matrix:")
print(confusion_matrix(densenet_true_classes, densenet_predictions))
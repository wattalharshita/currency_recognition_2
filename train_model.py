import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
import matplotlib.pyplot as plt




# Path to dataset folder
base_dir = os.path.join(os.getcwd(), 'dataset')

# Parameters
img_height = 180
img_width = 180
batch_size = 4  # keep it small for now

# Load dataset
datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)

model = Sequential([
    Input(shape=(224, 224, 3)),                      # Accepts images of size 224x224x3
    Conv2D(32, (3, 3), activation='relu'),           # First Conv layer
    MaxPooling2D(pool_size=(2, 2)),                  # Downsample

    Conv2D(64, (3, 3), activation='relu'),           # Second Conv layer
    MaxPooling2D(pool_size=(2, 2)),                  # Downsample again

    Flatten(),                                       # Flatten the feature map
    Dense(128, activation='relu'),                   # Fully connected layer
    Dense(6, activation='softmax')                   # Output layer for 6 classes
])

val_data = datagen.flow_from_directory(
    base_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

# Simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')  # 2 classes: real and fake
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
train_data = ImageDataGenerator(rescale=1./255).flow_from_directory(
    'train_model/dataset',  # Path to your training data folder
    target_size=(224, 224),  # Resize all images to 224x224
    batch_size=32,  # Process 32 images at a time
    class_mode='categorical'  # Because we're working on multi-class classification
)
# Save model
model.save("currency_model.h5")

print("✅ Model trained and saved as 'currency_model.h5'")

import os
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

# Set random seed for reproducibility
tf.random.set_seed(42)

# Function to view random images from the dataset
def view_random_image(target_dir, target_class):
    target_folder = os.path.join(target_dir, target_class)
    random_image = random.sample(os.listdir(target_folder), 1)[0]
    img_path = os.path.join(target_folder, random_image)
    
    img = mpimg.imread(img_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.title(target_class)
    plt.axis("off")
    
    print(f"Image shape: {img.shape}")
    return img

# Data paths
train_dir = "data/train/"
test_dir = "data/test/"

# Create data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(rescale=1./255)

# Create data flows
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(450, 450),
    batch_size=16,
    class_mode='categorical',
    shuffle=True
)

valid_data = valid_datagen.flow_from_directory(
    test_dir,
    target_size=(450, 450),
    batch_size=16,
    class_mode='categorical',
    shuffle=False
)

# Calculate steps per epoch
steps_per_epoch = train_data.samples // train_data.batch_size
validation_steps = valid_data.samples // valid_data.batch_size

# Custom callback for monitoring training
class DataMonitorCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"\nEpoch {epoch+1} complete")
        print(f"Training accuracy: {logs.get('accuracy'):.4f}")
        print(f"Validation accuracy: {logs.get('val_accuracy'):.4f}")

# Create the model
model = Sequential([
    Conv2D(32, 3, activation='relu', input_shape=(450, 450, 3)),
    MaxPool2D(2),
    Conv2D(64, 3, activation='relu'),
    MaxPool2D(2),
    Conv2D(64, 3, activation='relu'),
    MaxPool2D(2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(6, activation='softmax')
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Model summary
model.summary()

# Callbacks
callbacks = [
    DataMonitorCallback(),
    ModelCheckpoint(
        'best_batik_model.h5',
        save_best_only=True,
        monitor='val_accuracy'
    )
]

# Train the model
try:
    history = model.fit(
        train_data,
        epochs=20,
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_data,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
except Exception as e:
    print(f"An error occurred during training: {e}")

# Function to plot training history
def plot_loss_curves(history):
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Plot training history
if 'history' in locals():
    plot_loss_curves(history)

# Function to make predictions
def predict_image(model, image_path):
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(450, 450)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0
    
    prediction = model.predict(img_array)
    class_names = list(train_data.class_indices.keys())
    predicted_class = class_names[tf.argmax(prediction[0])]
    confidence = tf.reduce_max(prediction[0])
    
    return predicted_class, confidence

# Example usage of prediction (uncomment to use)
# test_image_path = "path/to/test/image.jpg"
# predicted_class, confidence = predict_image(model, test_image_path)
# print(f"Predicted class: {predicted_class}")
# print(f"Confidence: {confidence:.2f}")
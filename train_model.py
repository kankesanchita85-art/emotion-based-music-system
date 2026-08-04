import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt

# Dataset paths
train_dir = "dataset/train"
test_dir = "dataset/test"

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest"
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

# Training Dataset
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48,48),
    color_mode="grayscale",
    batch_size=32,
    class_mode="categorical",
    shuffle=True
)

# Validation Dataset
validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48,48),
    color_mode="grayscale",
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

# Compute Class Weights
labels = train_generator.classes

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)

class_weights = dict(enumerate(class_weights))
# ==========================
# Improved CNN Architecture
# ==========================

model = Sequential()

# Block 1
model.add(Conv2D(
    32,
    (3,3),
    padding="same",
    activation="relu",
    kernel_regularizer=l2(0.0005),
    input_shape=(48,48,1)
))
model.add(BatchNormalization())

model.add(Conv2D(
    32,
    (3,3),
    padding="same",
    activation="relu",
    kernel_regularizer=l2(0.0005)
))
model.add(BatchNormalization())

model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.20))


# Block 2
model.add(Conv2D(
    64,
    (3,3),
    padding="same",
    activation="relu",
    kernel_regularizer=l2(0.0005)
))
model.add(BatchNormalization())

model.add(Conv2D(
    64,
    (3,3),
    padding="same",
    activation="relu",
    kernel_regularizer=l2(0.0005)
))
model.add(BatchNormalization())

model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.30))


# Block 3
model.add(Conv2D(
    128,
    (3,3),
    padding="same",
    activation="relu",
    kernel_regularizer=l2(0.0005)
))
model.add(BatchNormalization())

model.add(Conv2D(
    128,
    (3,3),
    padding="same",
    activation="relu",
    kernel_regularizer=l2(0.0005)
))
model.add(BatchNormalization())

model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.40))


# Block 4
model.add(Conv2D(
    256,
    (3,3),
    padding="same",
    activation="relu",
    kernel_regularizer=l2(0.0005)
))
model.add(BatchNormalization())

model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.40))


# Fully Connected Layer
model.add(Flatten())

model.add(Dense(
    512,
    activation="relu",
    kernel_regularizer=l2(0.0005)
))
model.add(BatchNormalization())
model.add(Dropout(0.50))

model.add(Dense(7, activation="softmax"))


# ==========================
# Compile Model
# ==========================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()
# ==========================
# Callbacks
# ==========================

early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=15,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "model/best_emotion_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# ==========================
# Train Model
# ==========================

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=100,
    class_weight=class_weights,
    callbacks=[
        early_stop,
        checkpoint,
        reduce_lr
    ]
)

# ==========================
# Save Final Model
# ==========================

model.save("model/emotion_model.keras")

# ==========================
# Evaluate Model
# ==========================

loss, accuracy = model.evaluate(
    validation_generator,
    verbose=1
)

print("\n==============================")
print(f"Validation Accuracy : {accuracy*100:.2f}%")
print("==============================")

# ==========================
# Accuracy Graph
# ==========================

plt.figure(figsize=(8,5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("CNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig("assets/accuracy_graph.png")
plt.close()

# ==========================
# Loss Graph
# ==========================

plt.figure(figsize=(8,5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("CNN Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig("assets/loss_graph.png")
plt.close()

print("\nTraining Completed Successfully!")
print("Graphs Saved Successfully!")
print("Model Saved Successfully!")
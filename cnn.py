import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import os


train_ds = tf.keras.utils.image_dataset_from_directory(
  "C:\\Users\\HP\\data\\",
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(64, 64),
  batch_size=32)

val_ds = tf.keras.utils.image_dataset_from_directory(
  "C:\\Users\\HP\\data\\",
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(64, 64),
  batch_size=32)

test_ds = tf.keras.utils.image_dataset_from_directory(
  "C:\\Users\\HP\\test\\",
  image_size=(64, 64),
  batch_size=32)

class_names = train_ds.class_names
print(class_names)

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break


model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='softmax'),
  tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max',
                                               patience=30, restore_best_weights=True, verbose=1)

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', mode='max',
                                              factor=0.1, patience=10, verbose=1)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
    verbose=2,
    callbacks=[early_stopping, reduce_lr]
)


res = model.evaluate(test_ds, batch_size=32)
print(res)

y_out = model.predict(test_ds, batch_size=32)

class_names = test_ds.class_names
print(class_names)

plt.figure(figsize=(10, 10))
for images, labels in test_ds.take(1):
    for i in range(10):
        ax = plt.subplot(3, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")


print(y_out)

for i in range(len(y_out)):
  if y_out[i] > 0.5:
    print('malicious')
  else:
    print('benign')

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib
import string
import random

def main():

    data_dir = pathlib.Path('data')
    num_classes = 2 # boys and girls
    # image_count = len(list(data_dir.glob('*/*.jpg')))
    # boys = list(data_dir.glob('boy/*'))
    # girls = list(data_dir.glob('girl/*'))

    epochs = 10
    batch_size = 32
    img_height = 100 #400
    img_width = 125 #500

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )
    # Note: May want to use Resizing layer logic in model instead of image_size above
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/Resizing

    class_names = train_ds.class_names

    # configure dataset for performance
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # augment data
    data_augmentation = keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomFlip("vertical"),
        layers.experimental.preprocessing.RandomRotation(1.0),
        layers.experimental.preprocessing.RandomZoom(0.3),
    ])

    # visualise augmentation
    plt.figure(figsize=(10, 10))
    for images, _ in train_ds.take(1):
      for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")
    plt.show()

    # create the model
    model = Sequential([
        data_augmentation,
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(256, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Dense(128, activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Dense(64, activation='relu'),
        layers.Flatten(input_shape=(img_height, img_width)),
        layers.Dense(num_classes, activation='softmax'),
        # layers.Conv2D(16, 3, padding='same', activation='relu'),
        # layers.MaxPooling2D(),
        # layers.Conv2D(32, 3, padding='same', activation='relu'),
        # layers.MaxPooling2D(),
        # layers.Conv2D(64, 3, padding='same', activation='relu'),
        # layers.MaxPooling2D(),
        # layers.Flatten(),
        # layers.Dense(128, activation='relu'),
        # layers.Dense(num_classes)
    ])

    # compile the model
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # train the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
    )

    # model summary
    model.summary()

    # visualise the training result
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig("./results/{}_{}_{}_{}_{}.png".format(
        epochs, batch_size, img_height, img_width,
        ''.join(random.choice(string.ascii_lowercase) for i in range(3))
    ))
    plt.show()


def get_img_stats():

    count = 0
    width = 0
    height = 0
    lowest_width = 999
    lowest_height = 999

    for gender in ['boy', 'girl']:

        data_dir = os.path.join('data', gender)
        for image in os.listdir(data_dir):
            im = PIL.Image.open(os.path.join(data_dir, image))
            w, h = im.size

            width += w
            if w < lowest_width:
                lowest_width = w
            height += h
            if h < lowest_height:
                lowest_height = h

            count += 1

    # get average width and height of all images
    w = width/count
    h = height/count
    print("avr w: {}, h: {}".format(w, h))

    # corrected avrg
    wc = w/500
    hc = h/wc
    print("avr wc: {}, hc: {}".format(w/wc, hc))

    # lowest widths / heights
    print("lowest width: {}, lowest height: {}".format(lowest_width, lowest_height))


if __name__ == '__main__':
    # get_img_stats()
    main()

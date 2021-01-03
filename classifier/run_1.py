import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib

import matplotlib.pyplot as plt

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
    img_height = 400
    img_width = 500

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

    # test set
    val_batches = tf.data.experimental.cardinality(val_ds)
    test_dataset = val_ds.take(val_batches // 5)
    val_ds = val_ds.skip(val_batches // 5)
    print('Number of validation batches: %d' % tf.data.experimental.cardinality(val_ds))
    print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))

    # Note: May want to use Resizing layer logic in model instead of image_size above
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/Resizing

    class_names = train_ds.class_names

    # configure dataset for performance
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    # standardize data
    # normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

    # augment data
    data_augmentation = keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomFlip("vertical"),
        layers.experimental.preprocessing.RandomRotation(1.0),
        layers.experimental.preprocessing.RandomZoom(0.3),
        layers.experimental.preprocessing.RandomContrast(0.3),
    ])

    # # visualise augmentation
    # plt.figure(figsize=(10, 10))
    # for images, _ in train_ds.take(1):
    #   for i in range(9):
    #     augmented_images = data_augmentation(images)
    #     ax = plt.subplot(3, 3, i + 1)
    #     plt.imshow(augmented_images[0].numpy().astype("uint8"))
    #     plt.axis("off")
    # plt.show()

    # rescale pixel values
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)

    # Create the base model from the pre-trained model MobileNet V2
    IMG_SHAPE = (img_height, img_width) + (3,)
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')

    # Convert each 160x160x3 image into a 5x5x1280
    image_batch, label_batch = next(iter(train_ds))
    feature_batch = base_model(image_batch)
    print(feature_batch.shape)

    # Feature extraction
    base_model.trainable = False
    base_model.summary()

    # Add classification head
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)
    print(feature_batch_average.shape)

    prediction_layer = tf.keras.layers.Dense(1)
    prediction_batch = prediction_layer(feature_batch_average)
    print(prediction_batch.shape)

    # Build the model
    inputs = tf.keras.Input(shape=(img_height, img_width, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    # Compile the model
    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()
    len(model.trainable_variables)

    # train the model
    initial_epochs = 10
    loss0, accuracy0 = model.evaluate(val_ds)
    print("initial loss: {:.2f}".format(loss0))
    print("initial accuracy: {:.2f}".format(accuracy0))

    history = model.fit(train_ds,
                    epochs=initial_epochs,
                    validation_data=val_ds)

    # unfreeze top layer
    base_model.trainable = True
    # Let's take a look to see how many layers are in the base model
    print("Number of layers in the base model: ", len(base_model.layers))

    # Fine-tune from this layer onwards
    fine_tune_at = 100

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
      layer.trainable =  False

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
              metrics=['accuracy'])
    model.summary()
    len(model.trainable_variables)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    # plt.show()

    # Continue training the model
    fine_tune_epochs = 10
    total_epochs =  initial_epochs + fine_tune_epochs

    history_fine = model.fit(train_ds,
                             epochs=total_epochs,
                             initial_epoch=history.epoch[-1],
                             validation_data=val_ds)

    # # create the model
    # model = Sequential([
    #     data_augmentation,
    #     # layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    #     # layers.Conv2D(16, 3, padding='same', activation='relu'),
    #     # layers.MaxPooling2D(),
    #     layers.Conv2D(32, 3, padding='same', activation='relu'),
    #     layers.MaxPooling2D(),
    #     layers.Conv2D(64, 3, padding='same', activation='relu'),
    #     layers.MaxPooling2D(),
    #     layers.Flatten(),
    #     layers.Dense(128, activation='relu'),
    #     layers.Dropout(0.5), # dropout added
    #     layers.Dense(num_classes, activation='softmax')
    # ])
    #
    # # compile the model
    # model.compile(
    #     optimizer='adam',
    #     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #     metrics=['accuracy']
    # )
    #
    #
    # # train the model
    # history = model.fit(
    #     train_ds,
    #     validation_data=val_ds,
    #     epochs=epochs,
    # )
    #
    # # model summary
    # model.summary()

    # visualise the training result
    acc += history_fine.history['accuracy']
    val_acc += history_fine.history['val_accuracy']

    loss += history_fine.history['loss']
    val_loss += history_fine.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.ylim([0.8, 1])
    plt.plot([initial_epochs-1,initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.ylim([0, 1.0])
    plt.plot([initial_epochs-1,initial_epochs-1],
             plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')

    plt.savefig("./results/tl_{}_{}_{}_{}_{}.png".format(
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

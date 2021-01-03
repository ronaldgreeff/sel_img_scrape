import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from keras import backend as K
from keras.metrics import BinaryAccuracy

import pathlib
import string
import random

# Using
# https://www.tensorflow.org/guide/keras/transfer_learning


def main():

    CLASSES = 2
    CHANNELS = 3
    BATCH_SIZE = 32
    IMG_SIZE = (160, 160)

    if K.image_data_format() == 'channels_first':
        IMG_SHAPE = (CHANNELS,) + IMG_SIZE
    else:
        IMG_SHAPE = IMG_SIZE + (CHANNELS,)

        data_dir = pathlib.Path('data/4')

    initial_epochs = 10
    fine_tune_epochs = 10
    total_epochs = initial_epochs + fine_tune_epochs # used in fine-tuning

    base_lr = 0.0001*10
    fine_tune_lr = 0.0001/10
    momentum = 0.9
    fine_tune_at = 100 # Fine-tune from this layer onwards

    metrics = [BinaryAccuracy()] # ['accuracy'])

    pltid = '{}{}_{}{}_i{}f{}_b{}f{}'.format(
        ''.join(random.choice(string.ascii_lowercase) for i in range(3)),
        'baseMobileNetV2',
        IMG_SIZE[0],
        BATCH_SIZE,
        initial_epochs,
        fine_tune_epochs,
        base_lr,
        fine_tune_lr,
    )

    # # Augmentation
    # train_datagen = ImageDataGenerator(
    #     rescale=1./255,
    #     rotation_range=40,
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     shear_range=0.2,
    #     zoom_range=0.2,
    #     validation_split=0.2, # important
    #     horizontal_flip=True,
    #     fill_mode='nearest',)
    #
    # test_datagen = ImageDataGenerator(
    #     rescale=1./255,
    #     validation_split=0.2,) # important
    #
    # train_dataset = train_datagen.flow_from_directory(
    #     batch_size=BATCH_SIZE,
    #     shuffle=True,
    #     target_size=IMG_SIZE,
    #     subset='training',
    #     class_mode='binary',
    # )
    # validation_dataset = test_datagen.flow_from_directory(
    #     batch_size=BATCH_SIZE,
    #     shuffle=True,
    #     target_size=IMG_SIZE,
    #     subset='validation',
    #     class_mode='binary',
    # )

    train_dataset  = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    validation_dataset = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    class_names = train_dataset.class_names

    val_batches = tf.data.experimental.cardinality(validation_dataset)
    test_dataset = validation_dataset.take(val_batches // 5)
    validation_dataset = validation_dataset.skip(val_batches // 5)

    print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
    print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))


    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    # # augmentation configuration for training
    # train_datagen = ImageDataGenerator(
    #     rescale=1./255,
    #     rotation_range=40,
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     shear_range=0.2,
    #     zoom_range=0.2,
    #     validation_split=0.2, # important
    #     horizontal_flip=True,
    #     fill_mode='nearest',)
    #
    # # augmentation configuration for testing:
    # test_datagen = ImageDataGenerator(
    #     rescale=1./255,
    #     validation_split=0.2,) # important

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
        tf.keras.layers.experimental.preprocessing.RandomFlip('vertical'),
        tf.keras.layers.experimental.preprocessing.RandomRotation(1),
        tf.keras.layers.experimental.preprocessing.RandomWidth(0.2),
        tf.keras.layers.experimental.preprocessing.RandomHeight(0.2),
        tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
    ])

    # for image, _ in train_dataset.take(1):
    #     plt.figure(figsize=(10, 10))
    #     first_image = image[0]
    #     for i in range(9):
    #         ax = plt.subplot(3, 3, i + 1)
    #         augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
    #         plt.imshow(augmented_image[0] / 255)
    #         plt.axis('off')
    #     plt.show()

    rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1) # alternative to mobilenet_v2 preprocessing
    # preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    ##############
    # BASE MOODEL
    ##############

    # Create the base model from the pre-trained convnets
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')

    image_batch, label_batch = next(iter(train_dataset))
    feature_batch = base_model(image_batch)
    print(feature_batch.shape)

    base_model.trainable = False
    base_model.summary()

    ######################
    # CLASSIFIER HEAD
    ######################

    # Add a classification head
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)
    print(feature_batch_average.shape)

    prediction_layer = tf.keras.layers.Dense(1)
    prediction_batch = prediction_layer(feature_batch_average)
    print(prediction_batch.shape)

    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = data_augmentation(inputs)
    x = rescale(x) # x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    ################
    # COMPILE BASE MODEL
    ################

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_lr, momentum=momentum),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=metrics)

    model.summary()
    len(model.trainable_variables)

    ################
    # TRAIN BASE MODEL
    ################

    loss0, accuracy0 = model.evaluate(validation_dataset)
    print("initial loss: {:.2f}".format(loss0))
    print("initial accuracy: {:.2f}".format(accuracy0))

    # history = model.fit(
    #     train_dataset,
    #     steps_per_epoch=train_dataset.samples // batch_size,
    #     epochs=initial_epochs,
    #     validation_data=validation_dataset,
    #     validation_steps=validation_dataset.samples // batch_size,)

    history = model.fit(train_dataset,
                        epochs=initial_epochs,
                        validation_data=validation_dataset)

    # plot results
    acc = history.history['binary_accuracy']
    val_acc = history.history['val_binary_accuracy']
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
    plt.savefig("./results/{}_{}.png".format(pltid, 'initial',))
    # plt.show()

    ##############
    # FINE TUNING
    ##############

    base_model.trainable = True
    # Let's take a look to see how many layers are in the base model
    print("Number of layers in the base model: ", len(base_model.layers))

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
      layer.trainable =  False

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.RMSprop(lr=fine_tune_lr, momentum=momentum),
              metrics=metrics)
    model.summary()
    len(model.trainable_variables)

    # history_fine = model.fit(
    #     train_dataset,
    #     steps_per_epoch=train_dataset.samples // batch_size,
    #     epochs=total_epochs,
    #     initial_epoch=history.epoch[-1],
    #     validation_data=validation_dataset,
    #     validation_steps=validation_dataset.samples // batch_size,)

    history_fine = model.fit(train_dataset,
                             epochs=total_epochs,
                             initial_epoch=history.epoch[-1],
                             validation_data=validation_dataset)

    acc += history_fine.history['binary_accuracy']
    val_acc += history_fine.history['val_binary_accuracy']
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
    plt.savefig("./results/{}_{}.png".format(pltid, 'fine_tune',))
    # plt.show()

    ############################
    # Evaluation and prediction
    ############################

    loss, accuracy = model.evaluate(test_dataset)
    print('Test accuracy :', accuracy)

    #Retrieve a batch of images from the test set
    image_batch, label_batch = test_dataset.as_numpy_iterator().next()
    predictions = model.predict_on_batch(image_batch).flatten()

    # Apply a sigmoid since our model returns logits
    predictions = tf.nn.sigmoid(predictions)
    predictions = tf.where(predictions < 0.5, 0, 1)

    outcomes = 'A:{}\nL:{}\nP:{}\nT:{}'.format(
        accuracy,
        loss,
        predictions.numpy(),
        label_batch)

    print('Predictions:\n', predictions.numpy())
    print('Labels:\n', label_batch)

    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image_batch[i].astype("uint8"))
        plt.title(class_names[predictions[i]])
        plt.axis("off")
        plt.figtext(0.99, 0.01, outcomes, horizontalalignment='right')
        plt.savefig("./results/{}_{}.png".format(pltid, 'eval',))
    # plt.show()


# build data generator
def build_data_gen(datagen, batch_size, data_dir, img_height, img_width, subset, class_mode):

    data_generator = datagen.flow_from_directory(batch_size=batch_size,
        directory=data_dir,
        shuffle=True,
        target_size=(img_height, img_width),
        subset=subset,
        class_mode=class_mode,)

    return data_generator

# fit data to a model
def fit_model_data(model, train_generator, validation_generator, epochs, batch_size):
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,)

    return model, history


if __name__ == '__main__':
    main()

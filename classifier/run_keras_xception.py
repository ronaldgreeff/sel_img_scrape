import string
import random
import os
import pathlib
import numpy as np
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential, load_model, Model
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.applications import Xception
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import math
from keras.metrics import BinaryAccuracy
from keras.optimizers import Nadam, SGD



# Steps outlined in
# https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/
# https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/

def main():

    CLASSES = 1
    CHANNELS = 3
    BATCH_SIZE = 16
    IMG_SIZE = (150, 150)

    if K.image_data_format() == 'channels_first':
        IMG_SHAPE = (CHANNELS,) + IMG_SIZE
    else:
        IMG_SHAPE = IMG_SIZE + (CHANNELS,)

    data_dir = pathlib.Path('data/3')

    initial_epochs = 50
    fine_tune_epochs = 50
    total_epochs = initial_epochs + fine_tune_epochs # used in fine-tuning

    base_lr = 0.0001
    initial_lr = base_lr*10
    fine_tune_lr = base_lr/10
    momentum = 0.9
    fine_tune_at = -11

    metrics = ['accuracy', BinaryAccuracy()][0]

    pltid = '{}_{}{}_lr{}'.format(
        # ''.join(random.choice(string.ascii_lowercase) for i in range(3)),
        'keras_xception',
        IMG_SIZE[0],
        BATCH_SIZE,
        initial_lr,
    )

    def initial_run():

        train_data = training_datagen().flow_from_directory(
            data_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary',
            shuffle=False,
            subset='training',)

        print(len(train_data.filenames))
        print(train_data.class_indices)
        print(len(train_data.class_indices))

        validation_data  = validation_datagen().flow_from_directory(
            data_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary',
            shuffle=False,
            subset='validation',)

        # validation_labels = validation_data.classes
        # validation_labels = to_categorical(validation_labels, num_classes=2)

        # build the VGG16 network
        base_model = Xception(weights='imagenet', include_top=False)
        base_model.trainable = False

        #add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(200, activation='elu')(x) # Dense(256, activation='relu', kernel_initializer='he_uniform')
        x = Dropout(0.4)(x) # 0.5
        x = Dense(170, activation='elu')(x)
        predictions = Dense(CLASSES, activation='hard_sigmoid')(x) # sigmoid

        model = Model(inputs=base_model.input, outputs=predictions)
        model.summary()

        model.compile(optimizer=Nadam(lr=initial_lr),
            loss='binary_crossentropy', metrics=metrics)

        history = model.fit(
            train_data,
            steps_per_epoch=train_data.samples // BATCH_SIZE,
            epochs=initial_epochs,
            validation_data=validation_data,
            validation_steps=validation_data.samples // BATCH_SIZE,)

        # val_acc = history.history['val_accuracy']
        # val_loss = history.history['val_loss']
        # (eval_loss, eval_accuracy) = model.evaluate(
        #     validation_data, validation_labels, batch_size=BATCH_SIZE, verbose=1)
        # print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
        # print("[INFO] Loss: {}".format(eval_loss))

        model.save_weights('models/{}initial_E{}LR{}.h5'.format(pltid, initial_epochs, initial_lr))
        model.save('models/{}_initial_model_E{}LR{}'.format(pltid, initial_epochs, initial_lr))

        graph_history(
            '{}_initial_E{}LR{}'.format(pltid, initial_epochs, initial_lr),
            history, initial_epochs)

    def fine_tune():

        train_data = training_datagen().flow_from_directory(
            data_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary',
            shuffle=False,
            subset='training',)

        print(len(train_data.filenames))
        print(train_data.class_indices)
        print(len(train_data.class_indices))

        validation_data  = validation_datagen().flow_from_directory(
            data_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary',
            shuffle=False,
            subset='validation',)

        nb_train_samples = len(train_data.filenames)

        model = load_model('models/{}_initial_model_E{}LR{}'.format(pltid, initial_epochs, initial_lr))
        model.load_weights('models/{}initial_E{}LR{}.h5'.format(pltid, initial_epochs, initial_lr))

        model.trainable = True
        for layer in model.layers[:fine_tune_at]:
            layer.trainable = False

        model.compile(loss='binary_crossentropy',
                      optimizer=SGD(lr=fine_tune_lr, momentum=momentum),
                      metrics=['accuracy'])

        model.summary()

        history = model.fit(
            train_data,
            steps_per_epoch=train_data.samples // BATCH_SIZE,
            epochs=total_epochs,
            validation_data=validation_data,
            validation_steps=validation_data.samples // BATCH_SIZE,)

        # val_acc = history.history['val_accuracy']
        # val_loss = history.history['val_loss']
        #
        # (eval_loss, eval_accuracy) = model.evaluate(
        #     validation_data, validation_labels, batch_size=BATCH_SIZE, verbose=1)
        #
        # print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
        # print("[INFO] Loss: {}".format(eval_loss))

        model.save_weights('models/{}finetuned_E{}LR{}.h5'.format(pltid, total_epochs, fine_tune_lr))
        model.save('models/{}_finetuned_model_E{}LR{}'.format(pltid, total_epochs, fine_tune_lr))

        graph_history(
            '{}_finetuned_E{}LR{}'.format(pltid, total_epochs, fine_tune_lr),
            history, total_epochs)

    initial_run()
    fine_tune()


def training_datagen():
    return ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=360,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        validation_split=0.2, # important
        horizontal_flip=True,
        fill_mode='nearest')

def validation_datagen():
    return ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.2,)

def graph_history(str, history, epochs):

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
    plt.savefig("./results/{}_{}.png".format(str, ''.join(random.choice(string.ascii_lowercase) for i in range(3))))
    # plt.show()


if __name__ == '__main__':
    main()

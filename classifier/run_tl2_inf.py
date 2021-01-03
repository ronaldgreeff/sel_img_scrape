# import os
# import pathlib
# import numpy as np
# import matplotlib.pyplot as plt
#
# from keras.preprocessing.image import ImageDataGenerator
# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D
# from keras.layers import Activation, Dropout, Flatten, Dense
# from keras import applications
from keras import backend as K

import os
import pathlib
# from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import (Conv2D, MaxPooling2D, Dense, Flatten, Dropout)
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

from keras.metrics import BinaryAccuracy

# Steps outlined in
# https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/
# https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/

# 555/555 [==============================] - 259s 466ms/step - loss: 0.6495 - accuracy: 0.6294 - val_loss: 0.6444 - val_accuracy: 0.6479

def main():

    CLASSES = 2
    CHANNELS = 3
    # data_dirs = ['dogs-vs-cats',]
    # data_dirs = ['ideal', '4',]# '3', '2', '1', '0']
    # data_dir = pathlib.Path('data/dogs-vs-cats')
    # data_dir = pathlib.Path('data/3')

    # dimensions of our images.
    img_width, img_height = 150, 150
    epochs = 10
    batch_size = 36

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    # augmentation configuration for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=360,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        validation_split=0.2, # important
        horizontal_flip=True,
        fill_mode='nearest',)

    # augmentation configuration for testing:
    test_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,)

    # train on ideal
    ################

    data_dir = pathlib.Path('data/ideal')

    train_generator = train_datagen.flow_from_directory(batch_size=batch_size,
                                                     directory=data_dir,
                                                     shuffle=True,
                                                     target_size=(img_height, img_width),
                                                     subset="training",
                                                     class_mode='binary',)

    validation_generator = test_datagen.flow_from_directory(batch_size=batch_size,
                                                     directory=data_dir,
                                                     shuffle=True,
                                                     target_size=(img_height, img_width),
                                                     subset="validation",
                                                     class_mode='binary',)

    model = define_model(input_shape, lr=0.0001, momentum=0.1)

    val_acc = 0.0
    val_loss = 0.0
    history = None

    while val_acc < 0.95 || val_loss > 0.5 : #&& val_loss > 0.1:

        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size,)

            val_acc = history.history['val_binary_accuracy']
            val_loss = history.history['val_loss']

    model.save_weights('models/tl2inf_ideal.h5')

    graph_history('tl2inf_ideal', history, epochs)


    data_dir = pathlib.Path('data/4')

    train_generator = train_datagen.flow_from_directory(batch_size=batch_size,
                                                     directory=data_dir,
                                                     shuffle=True,
                                                     target_size=(img_height, img_width),
                                                     subset="training",
                                                     class_mode='binary',)

    validation_generator = test_datagen.flow_from_directory(batch_size=batch_size,
                                                     directory=data_dir,
                                                     shuffle=True,
                                                     target_size=(img_height, img_width),
                                                     subset="validation",
                                                     class_mode='binary',)

    opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[BinaryAccuracy()])

    val_acc = 0.0
    val_loss = 0.0

    while val_acc < 0.95 || val_loss > 0.5 : #&& val_loss > 0.1:

        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size,)

            val_acc = history.history['val_binary_accuracy']
            val_loss = history.history['val_loss']

    model.save_weights('models/tl2inf_d4.h5')


    graph_history('tl2inf_d4', history, epochs)


# define cnn model
def define_model(input_shape, lr, momentum):
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=input_shape))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = SGD(lr=lr, momentum=momentum)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[BinaryAccuracy()])
	return model


def graph_history(str, history, epochs):

    import string
    import random
    import matplotlib.pyplot as plt

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

if __name__ == '__main__':
    main()

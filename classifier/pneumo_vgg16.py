import os
import pathlib
import string
import random
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras import backend as K
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
# import numpy as np
from keras import applications, Model
from keras.optimizers import Adam, SGD
# from keras.utils import to_categorical


def main():

    img_width, img_height = 128, 128
    epochs = 1

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    title = 'pnuemonia_ideal'
    data_dir = pathlib.Path('data/ideal')
    split = 0.5 #0.2
    batch_size = 2#36
    epochs = epochs

    train_generator, validation_generator = get_datagens(data_dir, split, batch_size, img_height, img_width)

    base_model = applications.VGG16(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(170, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.summary()

    # opt = SGD(lr=1.0, momentum=0.9)
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        verbose=2,
        )

    graph_history(title, history, epochs)
    model.save_weights('models/{}.h5'.format(title))
    model.save('models/{}'.format(title))

    #########################################################################################################

    data_dir = pathlib.Path('data/3')

    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

    split = 0.2
    batch_size = 36
    epochs = epochs

    train_generator, validation_generator = get_datagens(data_dir, split, batch_size, img_height, img_width)

    depth = 7
    layers = 5
    layer_n = 4

    while depth:
        title = 'pnuemonia_3_{}'.format(depth)
        print(title)

        x = (7 - depth) * layer_n

        for layer in model.layers[:-1*(layers+x)]:
            layer.trainable = True

        if depth == 7:
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        if depth == 6:
            model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
        if depth < 6:
            model.compile(optimizer=Adam(lr=0.00001), loss='binary_crossentropy', metrics=['accuracy'])

        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size,
            # verbose=2,
            )

        graph_history(title, history, epochs)
        model.save_weights('models/{}.h5'.format(title))
        model.save('models/{}'.format(title))

        depth -= 1


    # title = 'pnuem3'
    # data_dir = pathlib.Path('data/3')
    # split = 0.2
    # batch_size = 36
    # epochs = epochs
    #
    # train_generator, validation_generator = get_datagens(data_dir, split, batch_size, img_height, img_width)
    #
    # # opt = SGD(lr=1.0, momentum=0.9)
    # model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    #
    # history = model.fit(
    #     train_generator,
    #     steps_per_epoch=train_generator.samples // batch_size,
    #     epochs=epochs,
    #     validation_data=validation_generator,
    #     validation_steps=validation_generator.samples // batch_size,
    #     # verbose=2,
    #     )
    #
    # graph_history(title, history, epochs)
    # model.save_weights('models/{}.h5'.format(title))
    # model.save('models/{}'.format(title))



# define cnn model
def get_model_0(input_shape):
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

    return model

def get_model_1(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    return model

def get_model_2(input_shape):
    model = Sequential()
    model.add(Conv2D(64, (5, 5), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.1))
    model.add(Conv2D(128, (4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    # model.add(Conv2D(512, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.02)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    return model

def get_datagens(data_dir, split, batch_size, img_height, img_width):
    # augmentation configuration for training

    # TODO: even number of samples per class

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=360,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        validation_split=split, # important
        horizontal_flip=True,
        fill_mode='nearest',)

    # augmentation configuration for testing:
    test_datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=split,)

    train_generator = train_datagen.flow_from_directory(batch_size=batch_size,
                                                     directory=data_dir,
                                                     shuffle=True,
                                                     target_size=(img_height, img_width),
                                                     subset="training",
                                                     class_mode="binary",
                                                     # color_mode="grayscale",
                                                     )

    validation_generator = test_datagen.flow_from_directory(batch_size=batch_size,
                                                     directory=data_dir,
                                                     shuffle=True,
                                                     target_size=(img_height, img_width),
                                                     subset="validation",
                                                     class_mode="binary",
                                                     # color_mode="grayscale",
                                                     )

    return train_generator, validation_generator


def graph_history(str, history, epochs):

    epochs_range = range(epochs)

    acc = history.history['accuracy']
    loss = history.history['loss']

    if ('val_accuracy' in history.history) and ('val_loss' in history.history):
        val_acc = history.history['val_accuracy']
        val_loss = history.history['val_loss']

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

    else:
        epochs_range = range(epochs)
        plt.figure(figsize=(8, 8))
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.title('Accuracy vs Loss')
        plt.savefig("./results/{}_{}.png".format(str, ''.join(random.choice(string.ascii_lowercase) for i in range(3))))


if __name__ == '__main__':
    main()

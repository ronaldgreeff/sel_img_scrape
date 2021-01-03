import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import applications
from keras import backend as K

# Steps outlined in
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
# and https://gist.github.com/fchollet/f35fbc80e066a49d65f1688a7e99f069

def main():

    CLASSES = 2
    CHANNELS = 3
    # data_dir = pathlib.Path('data/dogs-vs-cats')
    data_dir = pathlib.Path('data/4')

    # dimensions of our images.
    img_width, img_height = 150, 150
    # train_data_dir = 'data/train'
    # validation_data_dir = 'data/validation'
    epochs = 50
    batch_size = 16

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    # model = Sequential()
    # model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(Conv2D(32, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(Conv2D(64, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(Flatten())
    # model.add(Dense(64))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1))
    # model.add(Activation('sigmoid'))
    #
    # model.compile(loss='binary_crossentropy',
    #               optimizer='rmsprop',
    #               metrics=['accuracy'])
    #
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
    #     fill_mode='nearest')
    #
    # # augmentation configuration for testing:
    # test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2,)
    #
    # # train and valdiation generators
    # train_generator = train_datagen.flow_from_directory(batch_size=batch_size,
    #                                                  directory=data_dir,
    #                                                  shuffle=True,
    #                                                  target_size=(img_height, img_width),
    #                                                  subset="training",
    #                                                  class_mode='binary')
    #
    # validation_generator = test_datagen.flow_from_directory(batch_size=batch_size,
    #                                                  directory=data_dir,
    #                                                  shuffle=True,
    #                                                  target_size=(img_height, img_width),
    #                                                  subset="validation",
    #                                                  class_mode='binary')
    #
    # history = model.fit(
    #     train_generator,
    #     steps_per_epoch=2000 // batch_size,
    #     epochs=epochs,
    #     validation_data=validation_generator,
    #     validation_steps=800 // batch_size)
    #
    # model.save_weights('models/first_try.h5')
    #
    # graph_history('data4', history, epochs)

    top_model_weights_path = 'models/bottleneck_fc_model.h5'

    def save_bottlebeck_features():
        datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2,)

        # build the VGG16 network
        model = applications.VGG16(include_top=False, weights='imagenet')

        # https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/
        # load model with include_top=False - output layers are not loaded

        train_generator = datagen.flow_from_directory(batch_size=batch_size,
                                                         directory=data_dir,
                                                         shuffle=False,
                                                         target_size=(img_height, img_width),
                                                         subset="training",
                                                         class_mode=None)
        bottleneck_features_train = model.predict(
            train_generator, train_generator.samples // batch_size)
        np.save(open('features/bottleneck_features_train.npy', 'wb'),
                bottleneck_features_train)

        validation_generator = datagen.flow_from_directory(batch_size=batch_size,
                                                         directory=data_dir,
                                                         shuffle=False,
                                                         target_size=(img_height, img_width),
                                                         subset="validation",
                                                         class_mode=None)
        bottleneck_features_validation = model.predict_generator(
            validation_generator, validation_generator.samples // batch_size)
        np.save(open('features/bottleneck_features_validation.npy', 'wb'),
                bottleneck_features_validation)


    def train_top_model():
        train_data = np.load(open('features/bottleneck_features_train.npy', 'rb'))
        train_labels = np.array(
            [0] * int(nb_train_samples / 2) + [1] * int(nb_train_samples / 2))

        validation_data = np.load(open('features/bottleneck_features_validation.npy', 'rb'))
        validation_labels = np.array(
            [0] * int(nb_validation_samples / 2) + [1] * int(nb_validation_samples / 2))

        model = Sequential()
        model.add(Flatten(input_shape=train_data.shape[1:]))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy', metrics=['accuracy'])

        h = model.fit(train_data, train_labels,
                  epochs=epochs,
                  batch_size=batch_size,
                  validation_data=(validation_data, validation_labels))
        model.save_weights(top_model_weights_path)

        graph_history('top_model', h, epochs)

    # save_bottlebeck_features()
    train_top_model()



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
    # plt.show()


def get_img_stats():

    import PIL

    count = 0
    width = 0
    height = 0
    lowest_width = 999
    lowest_height = 999

    for gender in ['boy', 'girl']:

        data_dir = os.path.join('ideal', gender)
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
    print("avr img dimensions for {}:\nw: {}, h: {}".format(data_dir, w, h))


if __name__ == '__main__':
    # get_img_stats()
    main()

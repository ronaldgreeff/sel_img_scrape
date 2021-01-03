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
from keras.metrics import BinaryAccuracy

import pathlib
import string
import random

# Steps outlined in
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
# and https://gist.github.com/fchollet/f35fbc80e066a49d65f1688a7e99f069

def main():

    CLASSES = 2
    CHANNELS = 3
    BATCH_SIZE = 32
    IMG_SIZE = (160, 160)

    if K.image_data_format() == 'channels_first':
        IMG_SHAPE = (CHANNELS,) + IMG_SIZE
    else:
        IMG_SHAPE = IMG_SIZE + (CHANNELS,)

    data_dir = pathlib.Path('data/3')

    initial_epochs = 50
    fine_tune_epochs = 1
    total_epochs = initial_epochs + fine_tune_epochs # used in fine-tuning

    base_lr = 0.0001*10 # *** 1e-4 in guide + momentum = 0.9
    fine_tune_lr = 0.0001/10
    momentum = 0.9
    fine_tune_at = 25 # Fine-tune from this layer onwards

    metrics = [BinaryAccuracy()] # ['accuracy'])

    pltid = '{}_{}{}_b{}f{}'.format(
        # ''.join(random.choice(string.ascii_lowercase) for i in range(3)),
        'keras1',
        IMG_SIZE[0],
        BATCH_SIZE,
        # initial_epochs,
        # fine_tune_epochs,
        base_lr,
        fine_tune_lr,
    )

    top_model_weights_path = 'models/{}.h5'.format(pltid)
    base_model = 'jyubaseVGG16_16032_i1f1_b0.001f1e-05'

    datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2,)
    train_generator = datagen.flow_from_directory(batch_size=BATCH_SIZE,
                                                     directory=data_dir,
                                                     shuffle=False,
                                                     target_size=IMG_SIZE,
                                                     subset="training",
                                                     class_mode=None)

    validation_generator = datagen.flow_from_directory(batch_size=BATCH_SIZE,
                                                     directory=data_dir,
                                                     shuffle=False,
                                                     target_size=IMG_SIZE,
                                                     subset="validation",
                                                     class_mode=None)

    def save_bottlebeck_features(train_generator, validation_generator):

        # build the VGG16 network
        model = applications.VGG16(include_top=False, weights='imagenet')

        # https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/
        # load model with include_top=False - output layers are not loaded

        bottleneck_features_train = model.predict(
            train_generator, train_generator.samples // BATCH_SIZE)
        np.save(open('features/{}_bottleneck_train.npy'.format(pltid), 'wb'),
                bottleneck_features_train)


        bottleneck_features_validation = model.predict_generator(
            validation_generator, validation_generator.samples // BATCH_SIZE)
        np.save(open('features/{}_bottleneck_validation.npy'.format(pltid), 'wb'),
                bottleneck_features_validation)


    def train_top_model(model, ts, vs):
        train_data = np.load(open('features/{}_bottleneck_train.npy'.format(pltid), 'rb'))
        train_labels = np.array(
            [0] * int(ts / 2) + [1] * int(ts / 2))

        validation_data = np.load(open('features/{}_bottleneck_validation.npy'.format(pltid), 'rb'))
        validation_labels = np.array(
            [0] * int(vs / 2) + [1] * int(vs / 2))

        model = Sequential()
        model.add(Flatten(input_shape=train_data.shape[1:]))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy', metrics=metrics)

        h = model.fit(train_data, train_labels,
                  epochs=initial_epochs,
                  batch_size=BATCH_SIZE,
                  validation_data=(validation_data, validation_labels))

        model.save_weights(top_model_weights_path)

        graph_history(pltid+'topmodel', h, initial_epochs)

    # save_bottlebeck_features(train_generator, validation_generator)
    train_top_model(model=base_model, ts=train_generator.samples, vs=1184)



def graph_history(str, history, epochs):

    import string
    import random
    import matplotlib.pyplot as plt

    # visualise the training result
    acc = history.history['binary_accuracy']
    val_acc = history.history['val_binary_accuracy']
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
    plt.savefig("./results/{}.png".format(str,))
    # plt.show()

if __name__ == '__main__':
    main()

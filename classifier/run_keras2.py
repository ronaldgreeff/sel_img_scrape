import string
import random
import matplotlib.pyplot as plt
import os
import pathlib
# from keras.utils import to_categorical
from keras.models import Model, Sequential
from keras.layers import (Conv2D, MaxPooling2D, Dense, Flatten, Dropout)
from keras.optimizers import SGD, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.metrics import BinaryAccuracy
import numpy as np

# Steps outlined in
# https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/
# https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/

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

    initial_epochs = 1
    fine_tune_epochs = 1
    total_epochs = initial_epochs + fine_tune_epochs # used in fine-tuning

    base_lr = 0.0001*10
    fine_tune_lr = 0.0001/10
    momentum = 0.9
    fine_tune_at = 25

    metrics = [BinaryAccuracy()] # ['accuracy'])

    pltid = '{}_{}{}_i{}f{}b{}f{}'.format(
        # ''.join(random.choice(string.ascii_lowercase) for i in range(3)),
        'keras2',
        IMG_SIZE[0],
        BATCH_SIZE,
        initial_epochs,
        fine_tune_epochs,
        base_lr,
        fine_tune_lr,
    )

    # augmentation configuration for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
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
        validation_split=0.2,) # important

    ##############
    # base model #
    ##############

    model = VGG16(include_top=False, weights='imagenet', input_shape=IMG_SHAPE)
    model.summary()

    model.trainable = False
    # for layer in model.layers:
    #     layer.trainable = False

    ##########################################
    model_layer = Flatten()(model.layers[-1].output)
    ##########################################

    ###############
    # build model #
    ###############

    model_layer = Dense(256, activation='relu', kernel_initializer='he_uniform')(model_layer)
    model_layer = Dropout(0.5)(model_layer)
    gender_output = Dense(CLASSES, activation='sigmoid')(model_layer)
    model = Model(inputs=model.inputs, outputs=gender_output)

    # train_generator = build_data_gen(train_datagen, BATCH_SIZE, data_dir, IMG_SIZE, 'training', None)
    # validation_generator = build_data_gen(test_datagen, BATCH_SIZE, data_dir, IMG_SIZE, 'validation', None)
    # model, history = fit_model_data(model, train_generator, validation_generator, initial_epochs, BATCH_SIZE)
    # model.save_weights('models/{}initial.h5'.format(str_id))
    # graph_history('{}'.format(pltid), history, initial_epochs)

    # save_bottlebeck_features(model, train_datagen, test_datagen, BATCH_SIZE, data_dir, IMG_SIZE, pltid)
    train_top_model(model, base_lr, initial_epochs, BATCH_SIZE, pltid)
    fine_tune_model(model, train_datagen, test_datagen, data_dir, IMG_SIZE, fine_tune_lr, momentum, pltid)


def save_bottlebeck_features(model, train_datagen, test_datagen, BATCH_SIZE, data_dir, IMG_SIZE, pltid):

    datagen = ImageDataGenerator(rescale=1. / 255)

     generator = datagen.flow_from_directory(
         train_data_dir,
         target_size=(img_width, img_height),
         batch_size=batch_size,
         class_mode=None,
         shuffle=False)

     nb_train_samples = len(generator.filenames)
     num_classes = len(generator.class_indices)

     predict_size_train = int(math.ceil(nb_train_samples / batch_size))

     bottleneck_features_train = model.predict_generator(
         generator, predict_size_train)

     np.save('bottleneck_features_train.npy', bottleneck_features_train)

     generator = datagen.flow_from_directory(
         validation_data_dir,
         target_size=(img_width, img_height),
         batch_size=batch_size,
         class_mode=None,
         shuffle=False)

     nb_validation_samples = len(generator.filenames)

     predict_size_validation = int(math.ceil(nb_validation_samples / batch_size))

     bottleneck_features_validation = model.predict_generator(
         generator, predict_size_validation)

     np.save('bottleneck_features_validation.npy', bottleneck_features_validation)

    # train_generator = build_data_gen(train_datagen, BATCH_SIZE, data_dir, IMG_SIZE, 'training', None)
    # validation_generator = build_data_gen(test_datagen, BATCH_SIZE, data_dir, IMG_SIZE, 'validation', None)
    #
    # bottleneck_features_train = model.predict(
    #     train_generator, train_generator.samples // BATCH_SIZE)
    # np.save(open('features/{}_bottleneck_train.npy'.format(pltid), 'wb'),
    #         bottleneck_features_train)
    #
    # bottleneck_features_validation = model.predict(
    #     validation_generator, validation_generator.samples // BATCH_SIZE)
    # np.save(open('features/{}_bottleneck_validation.npy'.format(pltid), 'wb'),
    #         bottleneck_features_validation)


def train_top_model(model, base_lr, initial_epochs, BATCH_SIZE, pltid):

    datagen_top = ImageDataGenerator(rescale=1./255)
    generator_top = datagen_top.flow_from_directory(
         train_data_dir,
         target_size=(img_width, img_height),
         batch_size=batch_size,
         class_mode='categorical',
         shuffle=False)

    nb_train_samples = len(generator_top.filenames)
    num_classes = len(generator_top.class_indices)

    # load the bottleneck features saved earlier
    train_data = np.load('bottleneck_features_train.npy')

    # get the class lebels for the training data, in the original order
    train_labels = generator_top.classes

    # convert the training labels to categorical vectors
    train_labels = to_categorical(train_labels, num_classes=num_classes)

    generator_top = datagen_top.flow_from_directory(
             validation_data_dir,
             target_size=(img_width, img_height),
             batch_size=batch_size,
             class_mode=None,
             shuffle=False)

     nb_validation_samples = len(generator_top.filenames)

     validation_data = np.load('bottleneck_features_validation.npy')

     validation_labels = generator_top.classes
     validation_labels = to_categorical(validation_labels, num_classes=num_classes)

    # train_data = np.load(open('features/{}_bottleneck_train.npy'.format(pltid), 'rb'))
    # # train_labels = np.array(
    # #     [0] * int(ts / 2) + [1] * int(ts / 2))
    #
    # validation_data = np.load(open('features/{}_bottleneck_validation.npy'.format(pltid), 'rb'))
    # # validation_labels = np.array(
    # #     [0] * int(vs / 2) + [1] * int(vs / 2))
    #
    # opt = RMSprop(lr=base_lr)#, momentum=momentum)
    # model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    # model.summary()
    #
    # history = model.fit(train_data, #train_labels,
    #           epochs=initial_epochs,
    #           batch_size=BATCH_SIZE,
    #           validation_data=validation_data,#(validation_data, validation_labels)
    #           )
    #
    # model.save_weights('models/{}topmodel.h5'.format(pltid))
    # graph_history(pltid+'topmodel', history, initial_epochs)


def fine_tune_model(model, train_datagen, test_datagen, data_dir, IMG_SIZE, fine_tune_lr, momentum, pltid):
    for layer in model.layers[25:]:
        layer.trainable = True

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=fine_tune_lr, momentum=momentum),
                  metrics=['accuracy'])

    train_generator = build_data_gen(train_datagen, BATCH_SIZE, data_dir, IMG_SIZE, 'training', 'binary')
    validation_generator = build_data_gen(test_datagen, BATCH_SIZE, data_dir, IMG_SIZE, 'validation', 'binary')

    model, history = fit_model_data(model, train_generator, validation_generator, initial_epochs, BATCH_SIZE)
    model.save_weights('models/{}fine.h5'.format(str_id))
    graph_history('{}'.format(pltid), history, initial_epochs)


# build data generator
def build_data_gen(datagen, batch_size, data_dir, IMG_SIZE, subset, class_mode):

    data_generator = datagen.flow_from_directory(batch_size=batch_size,
        directory=data_dir,
        shuffle=True,
        target_size=IMG_SIZE,
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

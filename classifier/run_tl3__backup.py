import os
import pathlib
# from keras.utils import to_categorical
from keras.models import Model, Sequential
from keras.layers import (Conv2D, MaxPooling2D, Dense, Flatten, Dropout)
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.applications.vgg16 import VGG16

# Steps outlined in
# https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/
# https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/

def main():

    CLASSES = 2
    CHANNELS = 3

    img_width, img_height = 150, 150

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

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

    model = VGG16(include_top=False, input_shape=input_shape)
    model.summary()

    for layer in model.layers:
        layer.trainable = False

	# add new classifier layers
    ################################################
    # moved this flatten layer to model.layers[-2]
    flat1 = Flatten()(model.layers[-1].output)
    # flat1 = model.layers[-1].output
    ################################################

    ###############
    # fetal model #
    ###############

    class_mode = 'categorical'
    classes = 6
    lr = 0.01

    fetal_class = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    fetal_output = Dense(classes, activation='sigmoid')(fetal_class)

    model = Model(inputs=model.inputs, outputs=fetal_output)
    opt = SGD(lr=lr, momentum=0.9)
    model.compile(optimizer=opt, loss='{}_crossentropy'.format(class_mode), metrics=['accuracy'])
    model.summary()

    # train fetal #
    ###############

    str_id = 'tl3_VG16_fetal'
    data_dir = pathlib.Path('data/fetal/fetal')
    batch_size = 36
    epochs = 1

    train_generator = build_data_gen(train_datagen, batch_size, data_dir, img_height, img_width, 'training', class_mode)
    validation_generator = build_data_gen(test_datagen, batch_size, data_dir, img_height, img_width, 'validation', class_mode)
    model, history = fit_model_data(model, train_generator, validation_generator, epochs, batch_size)
    model.save_weights('models/{}.h5'.format(str_id))
    graph_history('{}'.format(str_id), history, epochs)

    # - loss: 1.3042 - accuracy: 0.5281 - val_loss: 0.6796 - val_accuracy: 0.7136

    # TODO: it might be a good idea to fine-tune here, then freeze these layers before
    # piping in the rest of the data

    for layer in model.layers:
        layer.trainable = False

    ################
    # binary model #
    ################

    class_mode = 'binary'
    classes = 1

    # binary layer 1
    binary_class_1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(model.layers[-2].output)
    # binary layer 2
    binary_class_2 = Dense(128, activation='relu', kernel_initializer='he_uniform')(binary_class_1)

    # flatten layer
    ###############
    # binary_class_2 = Flatten()(binary_class_2)

    gender_output = Dense(classes, activation='sigmoid')(binary_class_2)
    model = Model(inputs=model.inputs, outputs=gender_output)

    # train binary 1 #
    ##################

    lr = 0.001

    opt = SGD(lr=lr, momentum=0.9)
    model.compile(optimizer=opt, loss='{}_crossentropy'.format(class_mode), metrics=['accuracy'])
    model.summary()

    str_id = 'tl3_VG16_fetal_ideal'
    data_dir = pathlib.Path('data/ideal')
    batch_size = 2
    epochs = 1

    train_generator = build_data_gen(train_datagen, batch_size, data_dir, img_height, img_width, 'training', class_mode)
    validation_generator = build_data_gen(test_datagen, batch_size, data_dir, img_height, img_width, 'validation', class_mode)
    model, history = fit_model_data(model, train_generator, validation_generator, epochs, batch_size)
    model.save_weights('models/{}.h5'.format(str_id))
    graph_history('{}'.format(str_id), history, epochs)

    # - loss: 0.9045 - accuracy: 0.2978 - val_loss: 0.6576 - val_accuracy: 0.5000

    # train binary 2 #
    ##################

    lr = 0.001

    opt = SGD(lr=lr, momentum=0.9)
    model.compile(optimizer=opt, loss='{}_crossentropy'.format(class_mode), metrics=['accuracy'])
    model.summary()

    str_id = 'tl3_VG16_fetal_ideal_data4'
    data_dir = pathlib.Path('data/4')
    batch_size = 36
    epochs = 1

    train_generator = build_data_gen(train_datagen, batch_size, data_dir, img_height, img_width, 'training', class_mode)
    validation_generator = build_data_gen(test_datagen, batch_size, data_dir, img_height, img_width, 'validation', class_mode)
    model, history = fit_model_data(model, train_generator, validation_generator, epochs, batch_size)
    model.save_weights('models/{}.h5'.format(str_id))
    graph_history('{}'.format(str_id), history, epochs)

    # - accuracy: 0.5735 - val_loss: 0.6785 - val_accuracy: 0.5918


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

import h5py
from keras.models import Model
from keras.layers import Input, Activation, Concatenate, Add
from keras.layers import Flatten, Dropout
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras import backend as K


def SqueezeNet(nb_classes, inputs=(224, 224, 3), input_tensor=None, include_top=True):
    """ Keras Implementation of SqueezeNet(arXiv 1602.07360)

    @param nb_classes: total number of final categories

    Arguments:
    inputs -- shape of the input images (channel, cols, rows)

    """

    if input_tensor is None:
        input_img = Input(shape=inputs)
    else:
        if not K.is_keras_tensor(input_tensor):
            input_img = Input(tensor=input_tensor, shape=inputs)
        else:
            input_img = input_tensor

    conv1 = Convolution2D(
        64, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        strides=(2, 2), padding='same', name='conv1',
        data_format="channels_last")(input_img)
    maxpool1 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool1', padding='same',
        data_format="channels_last")(conv1)
    fire2_squeeze = Convolution2D(
        16, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire2_squeeze',
        data_format="channels_last")(maxpool1)
    fire2_expand1 = Convolution2D(
        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire2_expand1',
        data_format="channels_last")(fire2_squeeze)
    fire2_expand2 = Convolution2D(
        64, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire2_expand2',
        data_format="channels_last")(fire2_squeeze)
    merge2 = Concatenate(axis=-1)([fire2_expand1, fire2_expand2])

    fire3_squeeze = Convolution2D(
        16, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire3_squeeze',
        data_format="channels_last")(merge2)
    fire3_expand1 = Convolution2D(
        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire3_expand1',
        data_format="channels_last")(fire3_squeeze)
    fire3_expand2 = Convolution2D(
        64, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire3_expand2',
        data_format="channels_last")(fire3_squeeze)
    merge3 = Concatenate(axis=-1)([fire3_expand1, fire3_expand2])
    maxpool3 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool3',
        data_format="channels_last")(merge3)


    fire4_squeeze = Convolution2D(
        32, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire4_squeeze',
        data_format="channels_last")(maxpool3)
    fire4_expand1 = Convolution2D(
        128, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire4_expand1',
        data_format="channels_last")(fire4_squeeze)
    fire4_expand2 = Convolution2D(
        128, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire4_expand2',
        data_format="channels_last")(fire4_squeeze)
    merge4 = Concatenate(axis=-1)([fire4_expand1, fire4_expand2])

    fire5_squeeze = Convolution2D(
        32, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire5_squeeze',
        data_format="channels_last")(merge4)
    fire5_expand1 = Convolution2D(
        128, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire5_expand1',
        data_format="channels_last")(fire5_squeeze)
    fire5_expand2 = Convolution2D(
        128, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire5_expand2',
        data_format="channels_last")(fire5_squeeze)
    merge5 = Concatenate(axis=-1)([fire5_expand1, fire5_expand2])
    maxpool5 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool5', padding='same',
        data_format="channels_last")(merge5)

    fire6_squeeze = Convolution2D(
        48, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire6_squeeze',
        data_format="channels_last")(maxpool5)
    fire6_expand1 = Convolution2D(
        192, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire6_expand1',
        data_format="channels_last")(fire6_squeeze)
    fire6_expand2 = Convolution2D(
        192, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire6_expand2',
        data_format="channels_last")(fire6_squeeze)
    merge6 = Concatenate(axis=-1)([fire6_expand1, fire6_expand2])

    fire7_squeeze = Convolution2D(
        48, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire7_squeeze',
        data_format="channels_last")(merge6)
    fire7_expand1 = Convolution2D(
        192, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire7_expand1',
        data_format="channels_last")(fire7_squeeze)
    fire7_expand2 = Convolution2D(
        192, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire7_expand2',
        data_format="channels_last")(fire7_squeeze)
    merge7 = Concatenate(axis=-1)([fire7_expand1, fire7_expand2])

    fire8_squeeze = Convolution2D(
        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire8_squeeze',
        data_format="channels_last")(merge7)
    fire8_expand1 = Convolution2D(
        256, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire8_expand1',
        data_format="channels_last")(fire8_squeeze)
    fire8_expand2 = Convolution2D(
        256, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire8_expand2',
        data_format="channels_last")(fire8_squeeze)
    merge8 = Concatenate(axis=-1)([fire8_expand1, fire8_expand2])

    fire9_squeeze = Convolution2D(
        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire9_squeeze',
        data_format="channels_last")(merge8)
    fire9_expand1 = Convolution2D(
        256, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire9_expand1',
        data_format="channels_last")(fire9_squeeze)
    fire9_expand2 = Convolution2D(
        256, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire9_expand2',
        data_format="channels_last")(fire9_squeeze)
    merge9 = Concatenate(axis=-1)([fire9_expand1, fire9_expand2])
    maxpool9 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool9', padding='same',
        data_format="channels_last")(merge9)

    fire10_squeeze = Convolution2D(
        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire10_squeeze',
        data_format="channels_last")(maxpool9)
    fire10_expand1 = Convolution2D(
        256, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire10_expand1',
        data_format="channels_last")(fire10_squeeze)
    fire10_expand2 = Convolution2D(
        256, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire10_expand2',
        data_format="channels_last")(fire10_squeeze)
    merge10 = Concatenate(axis=-1)([fire10_expand1, fire10_expand2])    

    if include_top:
        fire10_dropout = Dropout(0.5, name='fire10_dropout')(merge10)
        conv11 = Convolution2D(
            nb_classes, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            padding='valid', name='conv10',
            data_format="channels_last")(fire10_dropout)
        global_avgpool11 =GlobalAveragePooling2D(data_format='channels_last')(conv11)
        softmax = Activation("softmax", name='softmax')(global_avgpool11)
        return Model(inputs=input_img, outputs=softmax)

    return Model(inputs=input_img, outputs=merge10)

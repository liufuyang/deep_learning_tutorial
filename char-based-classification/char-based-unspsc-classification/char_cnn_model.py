from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam
from keras.layers import Input, Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.initializers import RandomNormal

import tensorflow as tf


def model(filter_kernels, dense_outputs, maxlen, vocab_size, filters, unit_output):
    initializer = RandomNormal(mean=0.0, stddev=0.05, seed=None)

    def binarize(x, sz=vocab_size):
        return tf.to_float(tf.one_hot(x, sz, on_value=1, off_value=0, axis=-1))

    def binarize_outshape(in_shape):
        return in_shape[0], in_shape[1], vocab_size
    # Define what the input shape looks like
    # inputs = Input(shape=(maxlen, vocab_size), name='input', dtype='float32')

    inputs = Input(shape=(maxlen,), dtype='int64')

    # Use a lambda layer to create a onehot encoding of a sequence of characters on the fly.
    # Holding one-hot encodings in memory is very inefficient.
    embedded = Lambda(binarize, output_shape=binarize_outshape)(inputs)

    # All the convolutional layers...
    conv = Convolution1D(filters=filters[0], kernel_size=filter_kernels[0], kernel_initializer=initializer,
                         padding='same', activation='relu')(embedded)
    conv = MaxPooling1D(pool_size=2)(conv)

    conv1 = Convolution1D(filters=filters[1], kernel_size=filter_kernels[1], kernel_initializer=initializer,
                          padding='same', activation='relu')(conv)
    conv1 = MaxPooling1D(pool_size=2)(conv1)

    conv2 = Convolution1D(filters=filters[2], kernel_size=filter_kernels[2], kernel_initializer=initializer,
                          padding='same', activation='relu')(conv1)

    conv3 = Convolution1D(filters=filters[3], kernel_size=filter_kernels[3], kernel_initializer=initializer,
                          padding='same', activation='relu')(conv2)

    conv4 = Convolution1D(filters=filters[4], kernel_size=filter_kernels[4], kernel_initializer=initializer,
                          padding='valid', activation='relu')(conv3)

    conv5 = Convolution1D(filters=filters[5], kernel_size=filter_kernels[5], kernel_initializer=initializer,
                          padding='valid', activation='relu')(conv4)
    conv5 = MaxPooling1D(pool_size=2)(conv5)
    conv5 = Flatten()(conv5)

    # Two dense layers with dropout of .5
    z = Dropout(0.5)(Dense(dense_outputs, activation='relu',)(conv5))
    z = Dropout(0.5)(Dense(dense_outputs, activation='relu',)(z))

    # Output dense layer with softmax activation
    pred = Dense(unit_output, activation='softmax', name='output',)(z)

    _model = Model(inputs=inputs, outputs=pred)

    sgd = SGD(lr=0.01, momentum=0.9)
    adam = Adam(lr=0.0005)
    _model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return _model

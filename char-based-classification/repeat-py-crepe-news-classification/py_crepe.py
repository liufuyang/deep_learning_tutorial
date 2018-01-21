from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam
from keras.layers import Input, Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Convolution1D, MaxPooling1D

import tensorflow as tf



def model(filter_kernels, dense_outputs, maxlen, vocab_size, filters,
          unit_output):
          
    def binarize(x, sz=vocab_size):
        return tf.to_float(tf.one_hot(x, sz, on_value=1, off_value=0, axis=-1))

    def binarize_outshape(in_shape):
        return in_shape[0], in_shape[1], vocab_size
    #Define what the input shape looks like
    #inputs = Input(shape=(maxlen, vocab_size), name='input', dtype='float32')
    
    inputs = Input(shape=(maxlen,), dtype='int64')
    
    # Use a lambda layer to create a onehot encoding of a sequence of characters on the fly. 
    # Holding one-hot encodings in memory is very inefficient.
    embedded = Lambda(binarize, output_shape=binarize_outshape)(inputs)

    #All the convolutional layers...
    conv = Convolution1D(filters=filters, kernel_size=filter_kernels[0],
                         padding='valid', activation='relu',
                         input_shape=(maxlen, vocab_size))(embedded)
    conv = MaxPooling1D(pool_size=3)(conv)

    conv1 = Convolution1D(filters=filters, kernel_size=filter_kernels[1],
                          padding='valid', activation='relu')(conv)
    conv1 = MaxPooling1D(pool_size=3)(conv1)

    conv2 = Convolution1D(filters=filters, kernel_size=filter_kernels[2],
                          padding='valid', activation='relu')(conv1)

    conv3 = Convolution1D(filters=filters, kernel_size=filter_kernels[3],
                          padding='valid', activation='relu')(conv2)

    conv4 = Convolution1D(filters=filters, kernel_size=filter_kernels[4],
                          padding='valid', activation='relu')(conv3)

    conv5 = Convolution1D(filters=filters, kernel_size=filter_kernels[5],
                          padding='valid', activation='relu')(conv4)
    conv5 = MaxPooling1D(pool_size=3)(conv5)
    conv5 = Flatten()(conv5)

    #Two dense layers with dropout of .5
    z = Dropout(0.5)(Dense(dense_outputs, activation='relu')(conv5))
    z = Dropout(0.5)(Dense(dense_outputs, activation='relu')(z))

    #Output dense layer with softmax activation
    pred = Dense(unit_output, activation='softmax', name='output')(z)

    model = Model(inputs=inputs, outputs=pred)

    sgd = SGD(lr=0.01, momentum=0.9)
    adam = Adam(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=adam,
                  metrics=['accuracy'])
    
    
    # inputs2 = Input(shape=(maxlen,), dtype='int64')
    # embedded2 = Lambda(binarize, output_shape=binarize_outshape)(inputs2)
    # dense2 = Dense(200, activation='relu')(embedded2)
    # pred2 = Dense(unit_output, activation='softmax')(dense2)
    # 
    # model2 = Model(inputs=inputs2, outputs=pred2)
    # model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
    
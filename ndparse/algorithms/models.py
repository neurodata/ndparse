"""
models
"""

__author__ = "mjp"


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization



def ciresan_n3(n=65, nOutput=2):
    """An approximation of the N3 network from [1].
    Note that we also made a few small modifications along the way
    (from Theano to caffe and now to tensorflow/keras).

    As of this writing, no serious attempt has been made to optimize
    hyperparameters or structure of this network.

    Parameters:
       n : The tile size (diameter) to use in the sliding window.
           Tiles are assumed to be square, hence only one parameter.
    
    [1] Ciresan et al 'Deep neural networks segment neuronal membranes in
        electron microscopy images,' NIPS 2012.
    """
    model = Sequential()
    
    # input: nxn images with 1 channel -> (1, n, n) tensors.
    # this applies 48 convolution filters of size 5x5 each.
    model.add(Convolution2D(48, 5, 5, border_mode='valid', input_shape=(1, n, n)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(BatchNormalization())  # note: we used LRN previously...
    
    model.add(Convolution2D(48, 5, 5))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(BatchNormalization())  # note: we used LRN previously...
    #model.add(Dropout(0.25))

    model.add(Convolution2D(48, 5, 5, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    
    model.add(Flatten())
    # Note: Keras does automatic shape inference.
    model.add(Dense(200))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))

    model.add(Dense(nOutput))  # use 2 for binary classification
    model.add(Activation('softmax'))

    return model


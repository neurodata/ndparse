"""
A simple example of training the weights of a DNN.
Customize as needed for your application.

Example usage:
    THEANO_FLAGS='floatX=float32,device=gpu3' nohup python kast_train_synapse.py &
"""


import sys, os, copy, logging, socket, time


import numpy as np


sys.path.append('..'); import ndparse as ndp


if __name__ == '__main__': 
    logger = logging.getLogger("train_synapse")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('[%(asctime)s:%(name)s:%(levelname)s]  %(message)s'))
    logger.addHandler(ch)

    # load data
    data = np.load('deep_learning_kasthuri_example_data.npz')
    X = data['Xtrain']
    Y = data['Ytrain_synapse']

    # reshape to have desired dimensions
    X = np.transpose(X, [2, 0, 1]).astype(np.float32)
    X = X[:,np.newaxis,:,:]
    Y = np.transpose(Y, [2, 0, 1]).astype(np.float32)

    print('train data shape:   %s' % str(X.shape))
    print('train labels shape: %s' % str(Y.shape))

    # split into train and validation
    train_slices = np.arange(50)
    valid_slices = np.arange(95,100)
    n_epochs = 30

    # do it
    tic = time.time()
    model = ndp.nddl.train_model(X[train_slices,...], 
                                 np.squeeze(Y[train_slices, ...]),
                                 X[valid_slices,...], 
                                 np.squeeze(Y[valid_slices, ...]),
                                 omitLabels=[-1],
                                 nEpochs=n_epochs, 
                                 outDir='./synapse_weights',
                                 log=logger)

    print("Time to train: %0.2f sec" % (time.time() - tic))


# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

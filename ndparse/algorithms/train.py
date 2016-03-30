"""  Trains a CNN for dense (i.e. per-pixel) classification problems.

 The typical usage for this script is to run from the command line
 (often with nohup or equivalent, since it takes a long time).  
 However, it is possible to call the train_model() function directly
 from within a python shell or ipython notebook if desired.

 Note that this script all image data volumes have dimensions:

 (1)         #slices x #channels x rows x colums

 and all label volumes have dimensions:

 (2)         #slices x rows x colums


 To set the gpu id (from the command line) use the THEANO_FLAGS 
 environment variable
"""

__author__ = "Mike Pekala"
__copyright__ = "Copyright 2016, JHU/APL"
__license__ = "Apache 2.0"


import sys, os
import time
import random
import argparse
import logging
import numpy as np
import pdb

from keras.optimizers import SGD

import emlib
import models as emm





def _xform_minibatch(X):
    """Implements synthetic data augmentation by randomly appling
    an element of the group of symmetries of the square to a single 
    mini-batch of data.

    The default set of data augmentation operations correspond to
    the symmetries of the square (a non abelian group).  The
    elements of this group are:

      o four rotations (0, pi/2, pi, 3*pi/4)
        Denote these by: R0 R1 R2 R3

      o two mirror images (about y-axis or x-axis)
        Denote these by: M1 M2

      o two diagonal flips (about y=-x or y=x)
        Denote these by: D1 D2

    This page has a nice visual depiction:
      http://www.cs.umb.edu/~eb/d4/


    Parameters: 
       X := Mini-batch data (#examples, #channels, rows, colums) 
    """

    def R0(X):
        return X  # this is the identity map

    def M1(X):
        return X[:,:,::-1,:]

    def M2(X): 
        return X[:,:,:,::-1]

    def D1(X):
        return np.transpose(X, [0, 1, 3, 2])

    def R1(X):
        return D1(M2(X))   # = rot90 on the last two dimensions

    def R2(X):
        return M2(M1(X))

    def R3(X): 
        return D2(M2(X))

    def D2(X):
        return R1(M1(X))


    symmetries = [R0, R1, R2, R3, M1, M2, D1, D2]
    op = random.choice(symmetries) 
        
    # For some reason, the implementation of row and column reversals, 
    #     e.g.      X[:,:,::-1,:]
    # break PyCaffe.  Numpy must be doing something under the hood 
    # (e.g. changing from C order to Fortran order) to implement this 
    # efficiently which is incompatible w/ PyCaffe.  
    # Hence the explicit construction of X2 with order 'C' below.
    #
    # Not sure this matters for Theano/Keras, but leave in place anyway.
    X2 = np.zeros(X.shape, dtype=np.float32, order='C') 
    X2[...] = op(X)

    return X2




def _train_one_epoch(model, X, Y, 
                     omitLabels=[], 
                     batchSize=100,
                     nBatches=sys.maxint,
                     log=None):
    """Trains the model for one epoch.
    """
    #----------------------------------------
    # Pre-allocate some variables & storage.
    #----------------------------------------
    nChannels, nRows, nCols = model.input_shape[1:4]
    assert(nRows == nCols)
    ste = emlib.SimpleTileExtractor(nRows, X, Y, omitLabels=omitLabels)

    # some variables we'll use for reporting progress    
    lastChatter = -2
    startTime = time.time()
    gpuTime = 0
    accBuffer = []
    lossBuffer = []

    #----------------------------------------
    # Loop over mini-batches
    #----------------------------------------
    it = emlib.stratified_interior_pixel_generator(Y, 0, batchSize,
                                                   omitLabels=omitLabels,
                                                   stopAfter=nBatches*batchSize) 

    for mbIdx, (Idx, epochPct) in enumerate(it): 
        Xi, Yi = ste.extract(Idx)

        # label-preserving data transformation (synthetic data generation)
        Xi = _xform_minibatch(Xi)

        assert(not np.any(np.isnan(Xi)))
        assert(not np.any(np.isnan(Yi)))

        # do training
        tic = time.time()
        loss, acc = model.train_on_batch(Xi, Yi, accuracy=True)
        gpuTime += time.time() - tic

        accBuffer.append(acc);  lossBuffer.append(loss)

        #----------------------------------------
        # Some events occur on regular intervals.
        # Address these here.
        #----------------------------------------
        elapsed = (time.time() - startTime) / 60.0
        if (lastChatter+2) < elapsed:  
            # notify progress every 2 min
            lastChatter = elapsed

            if len(accBuffer) < 10:
                recentAcc = np.mean(accBuffer)
                recentLoss = np.mean(lossBuffer)
            else:
                recentAcc = np.mean(accBuffer[-10:])
                recentLoss = np.mean(lossBuffer[-10:])

            if log:
                log.info("  just completed mini-batch %d" % mbIdx)
                log.info("  we are %0.2g%% complete with this epoch" % (100.*epochPct))
                log.info("  recent accuracy, loss: %0.2f, %0.2f" % (recentAcc, recentLoss))
                fracGPU = (gpuTime/60.)/elapsed
                log.info("  pct. time spent on CNN ops.: %0.2f%%" % (100.*fracGPU))
                log.info("")

    # return statistics
    return accBuffer, lossBuffer



def _evaluate(model, X, Y, omitLabels=[], batchSize=100, log=None):
    """Evaluate model on held-out data.  Here, used to periodically
    report performance on validation data.
    """
    #----------------------------------------
    # Pre-allocate some variables & storage.
    #----------------------------------------
    nChannels, tileRows, tileCols = model.input_shape[1:4]
    tileRadius = int(tileRows/2)
    ste = emlib.SimpleTileExtractor(tileRows, X)

    numClasses = model.output_shape[-1]
    [numZ, numChan, numRows, numCols] = X.shape
    Prob = np.nan * np.ones([numZ, numClasses, numRows, numCols],
                            dtype=np.float32)

    #----------------------------------------
    # Loop over mini-batches
    #----------------------------------------
    it = emlib.interior_pixel_generator(X, tileRadius, batchSize)

    for mbIdx, (Idx, epochPct) in enumerate(it): 
        n = Idx.shape[0]         # may be < batchSize on final iteration
        Xi = ste.extract(Idx)
        prob = model.predict_on_batch(Xi)
        Prob[Idx[:,0], :, Idx[:,1], Idx[:,2]] = prob[0][:n,:]

    # Evaluate accuracy only on the subset of pixels that:
    #   o were actually provided to the CNN (not downsampled)
    #   o have a label that should be evaluated
    #
    # The mask tensor M will indicate which pixels to consider.
    M = np.all(np.isfinite(Prob), axis=1)
    for om in omitLabels:
        M[Y==om] = False
    Yhat = np.argmax(Prob, axis=1)  # probabilities -> class labels
    acc = 100.0 * np.sum(Yhat[M] == Y[M]) / np.sum(M)

    return Prob, acc



def train_model(Xtrain, Ytrain,
                Xvalid, Yvalid,
                trainSlices=[],
                validSlices=[],
                omitLabels=[],
                modelName='ciresan_n3',
                learnRate0=0.01,
                weightDecay=1e-6,
                momentum=0.9,
                maxMbPerEpoch=sys.maxint,
                nEpochs=30,
                log=None,
                outDir=None):
    """Trains a CNN using Keras.

    Some of the key parameters include:

      Xtrain, Ytrain : Tensors of features and per-pixel class labels with
                       dimensions as specified in (1),(2)
      Xvalid, Yalid :  Tensors of features and per-pixel class labels with
                       dimensions as specified in (1),(2).  Presumed to
                       be held-out data (i.e. disjoint from X/Ytrain)

      trainSlices   : A list of slice indices to include in training
                      (or [] to use all the data)
      validSlices   : A list of slice indices to include in validation
                      (or [] to use all the data)
      omitLabels    : A list of class labels whose corresponding pixel data
                      should be omitted from train and test.  If [], uses
                      all data.

      maxMbPerEpoch : The maximum number of minibatches to run in each
                      epoch (default is to process entire data volume
                      each epoch).

      log           : a logging object (for reporting status)
      outDir        : if not None, a directory where model weights
                      will be stored (highly recommended)
    """
    if not outDir: 
        if log: log.warning('No output directory specified - are you sure this is what you want?')
    elif not os.path.exists(outDir): 
        os.makedirs(outDir)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # preprocess data
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # discard unneeded slices (if any)
    if trainSlices:
        Xtrain = Xtrain[trainSlices,:,:,:]
        Ytrain = Ytrain[trainSlices,:,:]

    if validSlices:
        Xvalid = Xvalid[validSlices,:,:,:]
        Yvalid = Yvalid[validSlices,:,:]

    # rescale features to live in [0 1]
    # XXX: technically, should probably use scale factors from
    #      train volume on validation data...
    Xtrain = emlib.rescale_01(Xtrain, perChannel=True)
    Xvalid = emlib.rescale_01(Xvalid, perChannel=True)

    # Remap class labels to consecutive natural numbers.
    # Note that any pixels that should be omitted from the 
    # analysis are mapped to -1 by this function.
    Ytrain = emlib.number_classes(Ytrain, omitLabels)
    Yvalid = emlib.number_classes(Yvalid, omitLabels)


    if log: 
        log.info('training volume dimensions:   %s' % str(Xtrain.shape))
        log.info('training values min/max:      %g, %g' % (np.min(Xtrain), np.max(Xtrain)))
        log.info('training class labels:        %s' % str(np.unique(Ytrain)))
        for yi in np.unique(Ytrain):
            cnt = np.sum(Ytrain == yi)
            log.info('    class %d has %d instances' % (yi, cnt))
        log.info('')
        log.info('validation volume dimensions: %s' % str(Xvalid.shape))
        log.info('validation values min/max:    %g, %g' % (np.min(Xvalid), np.max(Xvalid)))
        log.info('validation class labels:      %s' % str(np.unique(Yvalid)))


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # create and configure CNN
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if log: log.info('creating CNN')
    model = getattr(emm, modelName)() 
    sgd = SGD(lr=learnRate0, decay=weightDecay, momentum=momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy', 
            class_mode='categorical', 
            optimizer=sgd)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Do training
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for epoch in range(nEpochs):
        if log: log.info('starting training epoch %d (of %d)' % (epoch, nEpochs));
        acc, loss = _train_one_epoch(model, Xtrain, Ytrain, 
                                     log=log,
                                     omitLabels=[-1,],
                                     nBatches=maxMbPerEpoch)

        if outDir: 
            # save a snapshot of current model weights
            weightFile = os.path.join(outDir, "weights_epoch_%03d.h5" % epoch) 
            if os.path.exists(weightFile): 
                os.remove(weightFile) 
            model.save_weights(weightFile)

            # also save accuracies (for diagnostic purposes)
            accFile = os.path.join(outDir, 'acc_epoch_%03d.npy' % epoch)
            np.save(accFile, acc)

        # Evaluate performance on validation data.
        if log: log.info('epoch %d complete. validating...' % epoch)
        Prob, acc = _evaluate(model, Xvalid, Yvalid, omitLabels=[-1,], log=log)
        if log: log.info('accuracy on validation data: %0.2f%%' % acc)

        if outDir: 
            estFile = os.path.join(outDir, "validation_epoch_%03d.npy" % epoch)
            np.save(estFile, Prob)


    if log: log.info('Finished!')
    return model



#-------------------------------------------------------------------------------
# Code for command-line interface
#-------------------------------------------------------------------------------

def _train_mode_args():
    """Parses command line arguments for training a CNN.

    Note that the variable names below need to align with the parameters
    in train_model() in order to have any effect.
    """

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--x-train', dest='emTrainFile', 
		    type=str, required=True,
		    help='Filename of the training volume (train mode)')
    parser.add_argument('--y-train', dest='labelsTrainFile', 
		    type=str, required=True,
		    help='Filename of the training labels (train mode)')
    parser.add_argument('--train-slices', dest='trainSlices', 
		    type=str, default='', 
		    help='(optional) limit to a subset of X/Y train')

    parser.add_argument('--x-valid', dest='emValidFile', 
		    type=str, required=True,
		    help='Filename of the validation volume (train mode)')
    parser.add_argument('--y-valid', dest='labelsValidFile', 
		    type=str, required=True,
		    help='Filename of the validation labels (train mode)')
    parser.add_argument('--valid-slices', dest='validSlices', 
		    type=str, default='', 
		    help='(optional) limit to a subset of X/Y validation')

    parser.add_argument('--omit-labels', dest='omitLabels', 
		    type=str, default='[-1,]', 
		    help='(optional) list of class labels to omit from training')
    
    parser.add_argument('--model', dest='model', 
		    type=str, default='ciresan_n3',
		    help='name of CNN model to use (python function)')
    parser.add_argument('--num-epochs', dest='nEpochs', 
		    type=int, default=30,
		    help='number of training epochs')
    parser.add_argument('--max-mb-per-epoch', dest='maxMbPerEpoch', 
		    type=int, default=sys.maxint,
		    help='maximum number of mini-batches to process each epoch')

    parser.add_argument('--out-dir', dest='outDir', 
		    type=str, default='', 
		    help='directory where the trained file should be placed')

    args = parser.parse_args()

    # Map strings into python objects.  
    # A little gross to use eval, but life is short.
    str_to_obj = lambda x: eval(x) if x else []
    
    args.trainSlices = str_to_obj(args.trainSlices)
    args.validSlices = str_to_obj(args.validSlices)
    args.omitLabels = str_to_obj(args.omitLabels)
    
    return args



def dict_subset(dictIn, keySubset):
    """Returns a subdictionary of 
    """
    return {k : dictIn[k] for k in dictIn if k in keySubset}


if __name__ == "__main__":
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # setup logging and output directory
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    logger = logging.getLogger('train_model')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('[%(asctime)s:%(name)s:%(levelname)s]  %(message)s'))
    logger.addHandler(ch)

    args = _train_mode_args()


    # Use command line args to override default args for train_model().
    # Note to self: the first co_argcount varnames are the 
    #               function's parameters.
    validArgs = train_model.__code__.co_varnames[0:train_model.__code__.co_argcount]
    cmdLineArgs = dict_subset(vars(args), validArgs)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # load training and validation volumes
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Xtrain = emlib.load_cube(args.emTrainFile, addChannel=True)
    Ytrain = emlib.load_cube(args.labelsTrainFile, addChannel=False)
        
    Xvalid = emlib.load_cube(args.emValidFile, addChannel=True)
    Yvalid = emlib.load_cube(args.labelsValidFile, addChannel=False)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # do it
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    model = train_model(Xtrain, Ytrain, Xvalid, Yvalid, log=logger, **cmdLineArgs)


# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

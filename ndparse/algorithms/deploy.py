"""Deploys a previously trained CNN on image data.

 See train.py for an example of how to train a CNN.

 Note that this script all image data volumes have dimensions:

 (1)         #slices x #channels x rows x colums


 Notes: 
   o Currently assumes input image data is single channel (i.e. grayscale).
   o Output probability values of -1 indicate pixels that were not 
     evaluated (e.g. due to boundary conditions or downsampling).
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
import scipy
import pdb

import emlib
import models as emm

from sobol_lib import i4_sobol_generate as sobol





def _downsample_mask(X, pct):
    """ Create a boolean mask indicating which subset of X should be 
    evaluated.
    """
    if pct < 1.0: 
        Mask = np.zeros(X.shape, dtype=np.bool)
        m = X.shape[-2]
        n = X.shape[-1]
        nToEval = np.round(pct*m*n).astype(np.int32)
        idx = sobol(2, nToEval ,0)
        idx[0] = np.floor(m*idx[0])
        idx[1] = np.floor(n*idx[1])
        idx = idx.astype(np.int32)
        Mask[:,:,idx[0], idx[1]] = True
    else:
        Mask = np.ones(X.shape, dtype=np.bool)

    return Mask



def _evaluate(model, X, log=None, batchSize=100, evalPct=1.0):
    """Evaluate model on held-out data.

    Returns:
      Prob : a tensor of per-pixel probability estimates with dimensions:
         (#layers, #classes, width, height)

    """
    #----------------------------------------
    # Pre-allocate some variables & storage.
    #----------------------------------------
    nChannels, tileRows, tileCols = model.input_shape[1:4]
    ste = emlib.SimpleTileExtractor(tileRows, X)

    lastChatter = -2
    startTime = time.time()

    # identify subset of volume to evaluate
    Mask = _downsample_mask(X, evalPct)
    if log: log.info('after masking, will evaluate %0.2f%% of data' % (100.0*np.sum(Mask)/Mask.size))

    # Create storage for class probabilities.
    # Note that we store all class probabilities, even if this
    # is a binary classification problem (in which case p(1) = 1 - p(0)).
    # We do this to support multiclass classification seamlessly.
    [numZ, numChan, numRows, numCols] = X.shape
    numClasses = model.output_shape[-1]
    Prob = -1*np.ones([numZ, numClasses, numRows, numCols], dtype=np.float32)

    #----------------------------------------
    # Loop over mini-batches
    #----------------------------------------
    # note: set tileRadius to 0 so we evaluate whole volume
    it = emlib.interior_pixel_generator(X, 0, batchSize, mask=Mask)

    for mbIdx, (Idx, epochPct) in enumerate(it): 
        n = Idx.shape[0] # may be < batchSize on final iteration
        Xi = ste.extract(Idx)
        prob = model.predict_on_batch(Xi)
        Prob[Idx[:,0], :, Idx[:,1], Idx[:,2]] = prob[0][:n,:]

        # notify user re. progress
        elapsed = (time.time() - startTime) / 60.0
        if (lastChatter+2) < elapsed:  
            lastChatter = elapsed
            if log: log.info("  last pixel %s (%0.2f%% complete)" % (str(Idx[-1,:]), 100.*epochPct))

    return Prob




def deploy_model(X, weightsFile,
                 log=None,
                 slices=[],
                 modelName='ciresan_n3',
                 evalPct=1.0,
                 outFile=None):
    """ Applies a previously trained CNN to new data.


      Xtrain        : Tensor of features with dimensions as specified in (1)
      trainSlices   : A list of slice indices to evalute 
                      (or [] to use all the data)
      log           : a logging object (for reporting status)
      outFile       : File name where class probabilities should be stored
      evalPct       : Fraction of volume to evalute; \in [0,1]
    """

    # Setup output file/dirctory
    if not outFile: 
        if log: log.warning('No output file specified - are you sure this is what you want?')
    elif not os.path.exists(os.path.dirname(outFile)):
        os.makedirs(os.path.dirname(outFile))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # preprocess data
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if slices:
        X = X[slices,:,:]

    # rescale features to live in [0, 1]
    X = emlib.rescale_01(X, perChannel=True)

    if log: log.info('X volume dimensions: %s' % str(X.shape))
    if log: log.info('X values min/max:    %g, %g' % (np.min(X), np.max(X)))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # initialize CNN
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if log: log.info('initializing CNN...')
    model = getattr(emm, modelName)() 
    model.compile(optimizer='sgd',   # not used, but required by keras
                  loss='categorical_crossentropy',
                  class_mode='categorical')
    model.load_weights(weightsFile)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Do it
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if log: log.info('evaluating volume...')
    Prob = _evaluate(model, X, log=log, evalPct=evalPct)
    if log: log.info('Complete!')

    if outFile: 
        np.save(outFile, Prob) 
        scipy.io.savemat(outFile.replace('.npy', '.mat'), {'P' : Prob})
        if log: log.info('Probabilites stored in file %s' % outFile)

    return Prob



#-------------------------------------------------------------------------------
# Code for command-line interface
#-------------------------------------------------------------------------------

def _deploy_mode_args():
    """Parameters for deploying a CNN.
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--x', dest='emFile', 
		    type=str, required=True,
		    help='Filename of volume to evaluate')

    parser.add_argument('--model', dest='modelName', 
		    type=str, default='ciresan_n3',
		    help='name of CNN model to use (python function)')

    parser.add_argument('--weight-file', dest='weightFile', 
		    type=str, required=True,
		    help='CNN weights to use')

    parser.add_argument('--slices', dest='slices', 
		    type=str, default='', 
		    help='(optional) subset of slices to evaluate')

    parser.add_argument('--eval-pct', dest='evalPct', 
		    type=float, default=1.0, 
		    help='(optional) Percent of pixels to evaluate (in [0,1])')

    parser.add_argument('--out-file', dest='outFile', 
		    type=str, required=True,
		    help='Ouput file name (will contain probability estimates)')

    args = parser.parse_args()

    if not os.path.exists(os.path.dirname(args.outFile)):
        os.makedirs(os.path.dirname(args.outFile))

    if not args.outFile.endswith('.npy'):
        args.outFile += '.npy'

    # Map strings into python objects.  
    # A little gross to use eval, but life is short.
    str_to_obj = lambda x: eval(x) if x else []
    
    args.slices = str_to_obj(args.slices)
    
    return args



if __name__ == "__main__":

    # setup logging 
    logger = logging.getLogger("deploy_model")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('[%(asctime)s:%(name)s:%(levelname)s]  %(message)s'))
    logger.addHandler(ch)


    args = _deploy_mode_args()

    # Use command line args to override default args for train_model().
    # Note to self: the first co_argcount varnames are the 
    #               function's parameters.
    from train import dict_subset
    validArgs = deploy_model.__code__.co_varnames[0:deploy_model.__code__.co_argcount]
    cmdLineArgs = dict_subset(vars(args), validArgs)

    # load data volume
    X = emlib.load_cube(args.emFile)

    # do it
    Prob = deploy_model(X, args.weightFile, log=logger, **cmdLineArgs)

# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

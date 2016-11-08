""" NDDL  Codes for Neuroscience Data Deep Learning.

This module contains code for solving dense classification problems on neuroscience image data.  In particular, it provides a simple implementation of the approach described in [1] using the Keras deep learning library [2].

Note that there are a number of more recent proposals for effective and computationally efficient ways to solve dense classification and/or segmentation problems.  The code in this module is more of a legacy approach at this point, provided as a convenience to other researchers in the community.

  REFERENCES:
   [1] Ciresan, Dan, et al. "Deep neural networks segment neuronal membranes in electron microscopy images." Advances in neural information processing systems. 2012.

   [2] http://keras.io/
"""

#__author__ = "mjp"
#__copyright__ = "Copyright 2016, JHU/APL"
#__license__ = "Apache 2.0"


import os, sys, re
import time
import random
import argparse
import logging
import pdb
from PIL import Image
import numpy as np
import scipy
from scipy.signal import convolve2d
from scipy.io import loadmat
from scipy.ndimage.morphology import distance_transform_edt as bwdist
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage.measurements import label as bwconncomp
import h5py

from .sobol_lib import i4_sobol_generate as sobol

#-------------------------------------------------------------------------------
# Functions for working with EM data files
#-------------------------------------------------------------------------------

def load_cube(dataFile, dtype='float32', addChannel=True):
    """ Loads a data volume.  This could be image data or per-pixel
    class labels.

    Uses the file extension to determine the underlying data format.
    Note that the Matlab data format currently assumes you saved using
    the -v7.3 flag (hdf5 under the hood).

      dataFile   : the full filename containing the data volume
      dtype      : data type that should be used to represent the data
      d4         : if true, adds the fourth channel dimension to the tensor
    """

    # Raw TIFF data
    if dataFile.lower().endswith('.tif') or dataFile.lower().endswith('.tiff'):
        X = load_tiff_data(dataFile, dtype)

    # Matlab data
    elif dataFile.lower().endswith('.mat'):
        # currently assumes matlab 7.3 format files - i.e. hdf
        #
        # Note: matlab uses fortran ordering, hence the permute/transpose here.
        d = h5py.File(dataFile, 'r')
        if len(list(d.keys())) > 1:
            raise RuntimeError('mat file has more than one key - not yet supported!')
        X = (list(d.values())[0])[:]
        X = np.transpose(X, (0,2,1))

    # Numpy file
    else:
        # otherwise assumpy numpy serialized object.
        X = np.load(dataFile)

    # No matter the source, make sure the type and dimensions are right.
    X = X.astype(dtype)
    if addChannel and X.ndim == 3:
        X = X[:, np.newaxis, :, :]

    return X


def load_tiff_data(dataFile, dtype='float32'):
    """ Loads data from a multilayer .tif file.

    dataFile := the tiff file name
    dtype    := data type to use for the returned tensor

    Returns result as a numpy tensor with dimensions (layers, width, height).
    """
    if not os.path.isfile(dataFile):
        raise RuntimeError('could not find file "%s"' % dataFile)

    # load the data from multi-layer TIF files
    dataImg = Image.open(dataFile)
    X = [];
    for ii in range(sys.maxsize):
        Xi = np.array(dataImg, dtype=dtype)
        if Xi.ndim == 2:
            Xi = Xi[np.newaxis, ...] # add slice dimension
        X.append(Xi)
        try:
            dataImg.seek(dataImg.tell()+1)
        except EOFError:
            break # this just means hit end of file (not really an error)

    X = np.concatenate(X, axis=0)  # list of 2d -> tensor
    return X




def number_classes(Yin, omitLabels=[]):
    """Remaps class labels to contiguous natural numbers starting at 0.
    In many frameworks (e.g. caffe) class labels are mapped to indices at
    the output of the CNN; hence this remapping.

    Any pixels that should be ignored will have class label of -1.
    """
    if Yin is None: return None

    yAll = np.sort(np.unique(Yin))
    yAll = [y for y in yAll if y not in omitLabels]

    Yout = -1*np.ones(Yin.shape, dtype=Yin.dtype)
    for yIdx, y in enumerate(yAll):
        Yout[Yin==y] = yIdx

    return Yout



def mirror_edges(X, nPixels):
    """Given a tensor X with dimension
         (z, c, row, col)

    produces a new tensor with dimensions
         (z, c, row+2*nPixels, row+2*nPixels)

    tensor with an "outer border" created by mirroring pixels along
    the outer border of X
    """
    assert(nPixels > 0)

    z,c,m,n = X.shape

    Xm = np.zeros((z, c, m+2*nPixels, n+2*nPixels), dtype=X.dtype)

    # the interior of Xm is just X
    Xm[:, :, nPixels:m+nPixels, nPixels:n+nPixels] = X

    # Note we do *not* replicate the pixel on the outer edge of the original image.
    for ii in range(z):
        for jj in range(c):
            # left edge
            Xm[ii,jj, :, 0:nPixels] = np.fliplr(Xm[ii,jj, :, (nPixels+1):(2*nPixels+1)])

            # right edge
            Xm[ii,jj, :, -nPixels:] = np.fliplr(Xm[ii,jj, :, (-2*nPixels-1):(-nPixels-1)])

            # top edge (fills in corners)
            Xm[ii,jj, 0:nPixels, :] = np.flipud(Xm[ii,jj, (nPixels+1):(2*nPixels+1), :])

            # bottom edge (fills in corners)
            Xm[ii,jj, -nPixels:, :] = np.flipud(Xm[ii,jj, (-2*nPixels-1):(-nPixels-1), :])

    return Xm



def rescale_01(X, perChannel=True):
    """Rescales all values to live in [0,1].
    """
    if not perChannel:
        xMin = np.min(X);  xMax = np.max(X)
        return (X - xMin) / (xMax - xMin)
    else:
        Xout = np.zeros(X.shape, dtype=X.dtype)
        for c in range(X.shape[1]):
            xMin = np.min(X[:,c,...])
            xMax = np.max(X[:,c,...])
            Xout[:,c,:,:] = (X[:,c,...] - xMin) / (xMax - xMin)
        return Xout



def interpolate_nn(X):
    """Simple nearest-neighbor based interplolation.
    Missing values will be replaced with the nearest non-missing value.
    Here, "missing values" are defined as those < 0.

      X : a tensor with dimensions:  (#slices, #classes, #rows, #cols)
    """

    Xout = np.zeros(X.shape, dtype=X.dtype)
    Xi = np.zeros((Xout.shape[-2], Xout.shape[-1]), dtype=X.dtype)

    for z in range(Xout.shape[0]):
        for c in range(Xout.shape[1]):
            Xi[:] = X[z,c,...]

            # interpolate, if needed
            if np.any(Xi < 0):
                #pct =  1.0*np.sum(Xi<0) / Xi.size
                dist, nn = bwdist(Xi<0, return_indices=True)
                Xi[Xi<0] = Xi[nn[0][Xi<0], nn[1][Xi<0]]

            Xout[z,c,...] = Xi

    return Xout



#-------------------------------------------------------------------------------
# Functions for extracting tiles from images
#-------------------------------------------------------------------------------

def _make_border_mask(sz, borderSize, omitSlices=[]):
    """ Creates a logical tensor of size

        (#slices, #rows, #colums)

    where 1/true is an "included" pixel, where "included" means
      - not within borderSize pixels the edge of the xy plane
      - not within a slice that is to be omitted.
    """
    [s,m,n] = sz

    bitMask = np.ones(sz, dtype=bool)
    bitMask[omitSlices,:,:] = 0

    if borderSize > 0:
        bitMask[:, 0:borderSize, :] = 0
        bitMask[:, (m-borderSize):m, :] = 0
        bitMask[:, :, 0:borderSize] = 0
        bitMask[:, :, (n-borderSize):n] = 0

    return bitMask



def stratified_interior_pixel_generator(Y, borderSize, batchSize,
                                        mask=None,
                                        omitSlices=[],
                                        omitLabels=[],
                                        stopAfter=-1):
    """An iterator over pixel indices with the property that pixels of
    different class labels are represented in equal proportions.

    Warning: this is fairly memory intensive (pre-computes the
    entire list of indices).
    An alternative (an approxmation) might have been random sampling...
    """
    [s,m,n] = Y.shape
    yAll = np.unique(Y)
    yAll = [y for y in yAll if y not in omitLabels]
    assert(len(yAll) > 0)

    # restrict the set of pixels under consideration.
    bitMask = _make_border_mask(Y.shape, borderSize, omitSlices)
    if mask is not None:
        bitMask = bitMask & mask

    # Determine how many instances of each class to report
    # (the minimum over the total number)
    cnt = [np.sum( (Y==y) & bitMask ) for y in yAll]
    cnt = min(cnt)

    # Stratified sampling
    Idx = np.zeros((0,3), dtype=np.int32)  # three columns because there are 3 dimensions in the tensor
    for y in yAll:
        tup = np.nonzero( (Y==y) & bitMask )
        Yi = np.column_stack(tup)
        np.random.shuffle(Yi)
        Idx = np.vstack((Idx, Yi[:cnt,:]))

    # one last shuffle to mix all the classes together
    np.random.shuffle(Idx)   # note: modifies array in-place

    # (optional) implement early stopping
    if (stopAfter > 0) and (stopAfter <= Idx.shape[0]):
        Idx = Idx[:stopAfter,...]

    # return in subsets of size batchSize
    for ii in range(0, Idx.shape[0], batchSize):
        nRet = min(batchSize, Idx.shape[0] - ii)
        yield Idx[ii:(ii+nRet)], (1.0*ii+nRet)/Idx.shape[0]



def region_sampling_pixel_generator(Y, borderSize, batchSize,
                                    dilateRadius=0,
                                    omitLabels=[]):
    """Samples points based on presumed cellular region structure.

    Zero values in Y are assumed to represent cell boundaries; non-zero values
    are presumed to be interior.  This function will produce a set of
    pixel indices with a roughly 50/50 split between cell interior and
    boundaries.  Furthermore, the interior values will be split
    approximately evenly across different cell interiors.

    Note that unlike other pixel generators, this one may draw repeat samples
    (in the case of cells with small interiors).

    This is not particularly quick to run; however, the time it takes is
    far outstripped by the time to run examples through the CNN...
    """
    [s,m,n] = Y.shape
    yAll = np.unique(Y)
    yAll = [y for y in yAll if y not in omitLabels]
    assert(0 in yAll)

    # restrict the set of pixels under consideration.
    Mask = _make_border_mask(Y.shape, borderSize)

    def dilate_by(Img, radius):
        "Dilates nonzero values in a 2d image"
        assert(Img.ndim == 2)
        if radius <= 0:
            return Img
        else:
            # structural element.
            W = np.zeros((2*radius+1, 2*radius+1), dtype=bool)
            W[radius, 0:] = True
            W[0:, radius] = True
            return binary_dilation(Img, W)

    def sample_indices(tup, nSamps):
        # This function exists mainly because of numpy's Byzantine
        # alternative to matlab's find().
        assert(len(tup)==2)
        n = len(tup[0])
        if n <= nSamps:
            rows = np.random.choice(np.arange(n), size=nSamps, replace=True)
        else:
            rows = np.random.choice(np.arange(n), size=nSamps, replace=False)
        Idx = np.array(tup).T
        return Idx[rows,:]


    # Determine pixels to sample on a per-slice basis
    AllIndices = np.zeros((0,3), dtype=np.int32)
    for z in range(Y.shape[0]):
        Yi = Y[z,...]
        if dilateRadius > 0:
            Yi = (Yi == 0)            # temporarily make 0 foreground
            Yi = dilate_by(Yi, dilateRadius)
            Yi = np.logical_not(Yi)   # revert to 1 as foreground

        Labels, nRegions = bwconncomp(Yi)

        # Determine how many pixels to sample from each region
        nBoundary = np.sum(Yi == 0)
        nInterior = np.sum(Yi != 0)
        if nInterior > nBoundary:
            nPerRegion = int(np.floor(1.0*nBoundary / nRegions))
        else:
            # TODO: this is actually fine, just need to downsample positive class.
            #       Implement this.
            raise RuntimeError('unexpected ratio of boundary to interior')

        # sample from cell interiors
        for regionId in np.unique(Labels):
            Bits = np.logical_and(Labels == regionId, Mask[z,...])
            indices = sample_indices(Bits.nonzero(), nPerRegion)
            zOnes = z*np.ones((indices.shape[0], 1), dtype=np.int32)
            indices = np.concatenate((zOnes, indices), axis=1)
            AllIndices = np.concatenate((AllIndices, indices), axis=0)

        # also (fully) sample cell boundaries (membranes)
        idx = (Y[z,...] == 0).nonzero()
        rows = idx[0][:,np.newaxis]
        cols = idx[1][:,np.newaxis]
        zOnes = z * np.ones((len(idx[0]),1), dtype=np.int32)
        indices = np.concatenate((zOnes, rows, cols), axis=1)
        AllIndices = np.concatenate((AllIndices, indices), axis=0)
        del idx, rows, cols, zOnes, indices

    # randomize example order
    np.random.shuffle(AllIndices)

    # return pixel indices one mini-batch at a time
    for ii in range(0, AllIndices.shape[0], batchSize):
        nRet = min(batchSize, AllIndices.shape[0] - ii)
        yield AllIndices[ii:(ii+nRet)], (1.0*ii+nRet)/AllIndices.shape[0]



def interior_pixel_generator(X, borderSize, batchSize,
                             mask=None,
                             omitSlices=[]):
    """An iterator over pixel indices in the interior of an image.

    Warning: this is fairly memory intensive (pre-computes the entire
    list of indices).

    Note: we could potentially speed up the process of extracting subtiles by
    creating a more efficient implementation; however, some simple timing tests
    indicated are spending orders of magnitude more time doing CNN operations
    so there is no pressing need to optimize tile extraction at the moment.

    Parameters:
      X          := An image tensor with *either* dimensions:
                       (#slices, #channels, width, height)
                    or
                       (#slices, width, height)

      borderSize := Specifies a border width - all pixels in this exterior
                    border will be excluded from the return value.

      batchSize  := The number of pixels that should be returned each iteration.

      mask       := a boolean tensor the same size as X where 0/false means
                    omit the corresponding pixel
    """
    if X.ndim == 4:
        [s,c,m,n] = X.shape
        # if the mask has a channel dimension, collapse it
        if mask is not None and mask.ndim == 4:
            mask = np.all(mask, axis=1)
    else:
        [s,m,n] = X.shape

    # Used to restrict the set of pixels under consideration.
    # Note that the number of channels plays no role here.
    bitMask = _make_border_mask([s,m,n], borderSize, omitSlices)

    if mask is not None:
        assert(np.all(mask.shape == bitMask.shape))
        bitMask = bitMask & mask

    Idx = np.column_stack(np.nonzero(bitMask))

    # return in subsets of size batchSize
    for ii in range(0, Idx.shape[0], batchSize):
        nRet = min(batchSize, Idx.shape[0] - ii)
        yield Idx[ii:(ii+nRet)], (1.0*ii+nRet)/Idx.shape[0]




class SimpleTileExtractor:
    """ Encapsulates the process of extracting tiles/windows centered at
    provided pixel locations.  This includes issues with mirroring
    to handle edge conditions.

    Makes a copy of X under the hood, so may be inappopriate for large
    data volumes.
    """

    def __init__(self, tileWidth, X, Y=None, omitLabels=[]):
        # The tile dimension must be odd and > 1
        assert(np.mod(tileWidth,2) == 1)
        assert(tileWidth > 1)

        tileRadius = int(tileWidth/2)
        self._X = mirror_edges(X, tileRadius)

        # Note: we do not (yet) know how many tiles will be in the batch.
        #       Defer actually allocating memory until later
        nChannels = X.shape[1]
        self._Xb = np.zeros([0, nChannels, tileWidth, tileWidth], dtype=np.float32)


        if (Y is not None) and (Y.size > 0):
            # Class labels will be indices into a one-hot vector; make sure
            # the labels are suitable for this purpose.
            yAll = np.unique(Y).astype(np.int32)
            yAll = [y for y in yAll if y not in omitLabels]
            nClasses = len(yAll)
            assert(np.min(yAll) == 0)
            assert(np.max(yAll) == nClasses-1)

            self._Yb = np.zeros([0, nClasses], dtype=np.float32)
            self._Y = Y
        else:
            # No class labels provided; this is fine.  The extract()
            # method will only return tiles from X.
            self._Yb = np.zeros([0,0])


    def extract(self, Idx):
        """
        Idx : an (n x 3) matrix, where the columns correspond to pixel
              depth, row and column.
        """
        assert(Idx.shape[1] == 3)
        n = Idx.shape[0]   # n := batch size
        tileWidth = self._Xb.shape[2]

        # (re)allocate memory, if needed
        # Note that if n is less than the previous batch size, old examples
        # will be reused.  This is intentional.
        if n > self._Xb.shape[0]:
            self._Xb = np.zeros( (n,) + self._Xb.shape[1:], dtype=np.float32)
            self._Yb = np.zeros( (n, self._Yb.shape[1]), dtype=np.float32)

        # Map pixel indices to tiles (and possibly class labels)
        for jj in range(n):
            # Note: Idx refers to coordinates in X, so we must account for
            #       the fact that _X has mirrored edges
            # Note: the code below is correcting for the
            a = Idx[jj,1]
            b = Idx[jj,1] + tileWidth
            c = Idx[jj,2]
            d = Idx[jj,2] + tileWidth

            self._Xb[jj, :, :, :] = self._X[ Idx[jj,0], :, a:b, c:d ]

            if self._Yb.size > 0:
                yj = int(self._Y[ Idx[jj,0], Idx[jj,1], Idx[jj,2] ])
                # store the class label as a 1 hot vector
                self._Yb[jj,:] = 0
                self._Yb[jj,yj] = 1

        if self._Yb.size:
            return self._Xb, self._Yb
        else:
            return self._Xb



#-------------------------------------------------------------------------------

def metrics(Y, Yhat, display=False):
    """
    PARAMETERS:
      Y    :  a numpy array of true class labels
      Yhat : a numpy array of estimated class labels (same size as Y)

    o Assumes any class label <0 should be ignored in the analysis.
    o Assumes all non-negative class labels are contiguous and start at 0.
      (so for binary classification, the class labels are {0,1})
    """
    assert(len(Y.shape) == 3)
    assert(len(Yhat.shape) == 3)

    # create a confusion matrix
    # yAll is all *non-negative* class labels in Y
    yAll = np.unique(Y);  yAll = yAll[yAll >= 0]
    C = np.zeros((yAll.size, yAll.size))
    for yi in yAll:
        est = Yhat[Y==yi]
        for jj in yAll:
            C[yi,jj] = np.sum(est==jj)

    # works for arbitrary # of classes
    acc = 1.0*np.sum(Yhat[Y>=0] == Y[Y>=0]) / np.sum(Y>=0)

    # binary classification metrics (only for classes {0,1})
    nTruePos = 1.0*np.sum((Y==1) & (Yhat==1))
    precision = nTruePos / np.sum(Yhat==1)
    recall = nTruePos / np.sum(Y==1)
    f1 = 2.0*(precision*recall) / (precision+recall)

    if display:
        for ii in range(C.shape[0]):
            print(('  class=%d    %s' % (ii, C[ii,:])))
        print(('  accuracy:  %0.3f' % (acc)))
        print(('  precision: %0.3f' % (precision)))
        print(('  recall:    %0.3f' % (recall)))
        print(('  f1:        %0.3f' % (f1)))

    return C, acc, precision, recall, f1


#-------------------------------------------------------------------------------
# Deep learning models
# In this module, we provide only one example deep learning model.
#-------------------------------------------------------------------------------

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

    from keras.optimizers import SGD
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Flatten
    from keras.layers import Convolution2D, MaxPooling2D
    from keras.layers.normalization import BatchNormalization


    model = Sequential()

    # input: nxn images with 1 channel -> (1, n, n) tensors.
    # this applies 48 convolution filters of size 5x5 each.
    model.add(Convolution2D(48, 5, 5, border_mode='valid', dim_ordering='th', input_shape=(1, n, n)))
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


#-------------------------------------------------------------------------------
#  Code for training a deep learning network
#-------------------------------------------------------------------------------

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
                     nBatches=sys.maxsize,
                     log=None):
    """Trains the model for one epoch.
    """
    #----------------------------------------
    # Pre-allocate some variables & storage.
    #----------------------------------------

    from keras.optimizers import SGD
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Flatten
    from keras.layers import Convolution2D, MaxPooling2D
    from keras.layers.normalization import BatchNormalization

    nChannels, nRows, nCols = model.input_shape[1:4]
    assert(nRows == nCols)
    ste = SimpleTileExtractor(nRows, X, Y, omitLabels=omitLabels)

    # some variables we'll use for reporting progress
    lastChatter = -2
    startTime = time.time()
    gpuTime = 0
    accBuffer = []
    lossBuffer = []

    #----------------------------------------
    # Loop over mini-batches
    #----------------------------------------
    it = stratified_interior_pixel_generator(Y, 0, batchSize,
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
        loss, acc = model.train_on_batch(Xi, Yi)
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



#def _evaluate(model, X, Y, omitLabels=[], batchSize=100, log=None):
#    """Evaluate model on held-out data.  Here, used to periodically
#    report performance on validation data.
#    """
#    #----------------------------------------
#    # Pre-allocate some variables & storage.
#    #----------------------------------------
#
#    from keras.optimizers import SGD
#    from keras.models import Sequential
#    from keras.layers import Dense, Dropout, Activation, Flatten
#    from keras.layers import Convolution2D, MaxPooling2D
#    from keras.layers.normalization import BatchNormalization
#
#    nChannels, tileRows, tileCols = model.input_shape[1:4]
#    tileRadius = int(tileRows/2)
#    ste = SimpleTileExtractor(tileRows, X)
#
#    numClasses = model.output_shape[-1]
#    [numZ, numChan, numRows, numCols] = X.shape
#    Prob = np.nan * np.ones([numZ, numClasses, numRows, numCols],
#                            dtype=np.float32)
#
#    #----------------------------------------
#    # Loop over mini-batches
#    #----------------------------------------
#    it = interior_pixel_generator(X, tileRadius, batchSize)
#
#    for mbIdx, (Idx, epochPct) in enumerate(it):
#        n = Idx.shape[0]         # may be < batchSize on final iteration
#        Xi = ste.extract(Idx)
#        prob = model.predict_on_batch(Xi)
#	# mjp: keras API change
#        #Prob[Idx[:,0], :, Idx[:,1], Idx[:,2]] = prob[0][:n,:]
#        Prob[Idx[:,0], :, Idx[:,1], Idx[:,2]] = prob[:n,:]
#
#    # Evaluate accuracy only on the subset of pixels that:
#    #   o were actually provided to the CNN (not downsampled)
#    #   o have a label that should be evaluated
#    #
#    # The mask tensor M will indicate which pixels to consider.
#    M = np.all(np.isfinite(Prob), axis=1)
#    for om in omitLabels:
#        M[Y==om] = False
#    Yhat = np.argmax(Prob, axis=1)  # probabilities -> class labels
#    acc = 100.0 * np.sum(Yhat[M] == Y[M]) / np.sum(M)
#
#    return Prob, acc



def train_model(Xtrain, Ytrain,
                Xvalid, Yvalid,
                trainSlices=[],
                validSlices=[],
                omitLabels=[],
                modelName='ciresan_n3',
                learnRate0=0.01,
                weightDecay=1e-6,
                momentum=0.9,
                maxMbPerEpoch=sys.maxsize,
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

    from keras.optimizers import SGD
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Flatten
    from keras.layers import Convolution2D, MaxPooling2D
    from keras.layers.normalization import BatchNormalization

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
    Xtrain = rescale_01(Xtrain, perChannel=True)
    Xvalid = rescale_01(Xvalid, perChannel=True)

    # Remap class labels to consecutive natural numbers.
    # Note that any pixels that should be omitted from the
    # analysis are mapped to -1 by this function.
    Ytrain = number_classes(Ytrain, omitLabels)
    Yvalid = number_classes(Yvalid, omitLabels)


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
    model = (globals()[modelName])()
    sgd = SGD(lr=learnRate0, decay=weightDecay, momentum=momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

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
        Prob, acc = _evaluate(model, Xvalid, Y=Yvalid, omitLabels=[-1,], log=log)
        if log: log.info('accuracy on validation data: %0.2f%%' % acc)

        if outDir:
            estFile = os.path.join(outDir, "validation_epoch_%03d.npy" % epoch)
            np.save(estFile, Prob)


    if log: log.info('Finished!')
    return model


#-------------------------------------------------------------------------------
#  Code for evaluating a model on new (test) data.
#-------------------------------------------------------------------------------

def dict_subset(dictIn, keySubset):
    """Returns a subdictionary of
    """
    return {k : dictIn[k] for k in dictIn if k in keySubset}




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



def _evaluate(model, X, batchSize=100, evalPct=1.0, 
              Y=None, omitLabels=[], log=None):
    """Evaluate model on a data volume.
    """
    #----------------------------------------
    # Pre-allocate some variables & storage.
    #----------------------------------------
    from keras.optimizers import SGD
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Flatten
    from keras.layers import Convolution2D, MaxPooling2D
    from keras.layers.normalization import BatchNormalization

    nChannels, tileRows, tileCols = model.input_shape[1:4]
    ste = SimpleTileExtractor(tileRows, X)

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
    it = interior_pixel_generator(X, 0, batchSize, mask=Mask)

    for mbIdx, (Idx, epochPct) in enumerate(it):
        n = Idx.shape[0]         # may be < batchSize on final iteration
        Xi = ste.extract(Idx)
        prob = model.predict_on_batch(Xi)
	# mjp: Keras API update
        #Prob[Idx[:,0], :, Idx[:,1], Idx[:,2]] = prob[0][:n,:]
        Prob[Idx[:,0], :, Idx[:,1], Idx[:,2]] = prob[:n,:]

        # notify user re. progress
        elapsed = (time.time() - startTime) / 60.0
        if (lastChatter+2) < elapsed:
            lastChatter = elapsed
            if log: log.info("  last pixel %s (%0.2f%% complete)" % (str(Idx[-1,:]), 100.*epochPct))

    #----------------------------------------
    # (optional) report accuracy
    #----------------------------------------
    if Y is not None:
        # evaluate subset of pixels that were (a) not downsampled
        # and (b) have a label that the caller cares about.
        # The mask M will address this.
        M = np.all(Prob >= 0, axis=1)
        for om in omitLabels:
            M[Y==om] = False
        Yhat = np.argmax(Prob, axis=1)  # probabilities -> class labels
        acc = 100.0 * np.sum(Yhat[M] == Y[M]) / np.sum(M)
        return Prob, acc
    else:
        return Prob




def fit(X, weightsFile,
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

    from keras.optimizers import SGD
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Flatten
    from keras.layers import Convolution2D, MaxPooling2D
    from keras.layers.normalization import BatchNormalization

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
    X = rescale_01(X, perChannel=True)

    if log: log.info('X volume dimensions: %s' % str(X.shape))
    if log: log.info('X values min/max:    %g, %g' % (np.min(X), np.max(X)))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # initialize CNN
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if log: log.info('initializing CNN...')
    model = (globals()[modelName])()
    model.compile(optimizer='sgd',   # not used, but required by keras
                  loss='categorical_crossentropy')
                  #class_mode='categorical')
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

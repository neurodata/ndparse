**ndparse** is the NeuroData Parse parent repository

ndparse:  NeuroData Parse
=========================

Installation
------------

You can either clone this repository and use it locally, or install from pypi:

~~~bash
pip install ndparse
~~~

Use this Python library to easily interface with NeuroData algorithms to
manually annotate data, use computer vision algorithms and deploy
across large neuroscience datasets.

This repo contains the code needed to train, evaluate, and deploy code for parsing volumes of NeuroData images. 

It contains the legacy code for manno and macho.  The current version of ndod is divided into the several major components required to parse neuroscience data at scale:

- **annotate**: manual annotation protocols and tools
- **algorithms**: big-data research algorithms to inform neuroscience
- **learn**: protocols for training and validation
- **deploy**: machine annotation for deployment and scaling.
- **utils**:  Utilities and convenience functions (e.g., plotting) for big data neuroscience

To just use one of these, say **mana**, in python you can (and should) type the following: `from ndparse import mana`

*This repo is under extremely active development during the first quarter of 2016.  The previous version of mano, macho and ndod code may be found in [ndod](https://github.com/neurodata/ndod).  The core code that is used for computer vision by the neurodata team will be transitioned to a pip installable python package in the next few weeks.  Stay tuned.*

On osx:
vigra 1.10.0 is needed (1.11.x currently causes a segfault)
blosc 1.2.3 is needed (1.3.x doesn't compile)

This is accounted for in the scripts below and doesn't affect performance.
~~~bash
pip install conda
conda create -n ndp -c ilastik ilastik-everything-but-tracking -y
source activate ndp
conda uninstall vigra -y
conda install --channel https://conda.anaconda.org/FlyEM vigra -y
conda install ipython notebook -y
conda install requests gcc blosc -y
pip install scikit-learn --upgrade
pip install scikit-image --upgrade
pip install mahotas
pip install blosc==1.2.0
pip install ndio
pip install ndparse
~~~


For ilastik processing:

~~~python

from time import time
t = time()
import ndparse as ndp
import numpy as np
import ndio.remote.neurodata as neurodata
nd = neurodata()
pad = 20
input_data = nd.get_cutout('bock11','image',22000-pad,23000+pad,22000-pad,23000+pad,3000-pad,3020+pad,resolution=1)
classifier = 'bock11_v0.ilp' #specify this!
probs = ndp.algorithms.run_ilastik_pixel(input_data, classifier,threads=4, ram=4000)
probs = ndp.utils.choose_channel_4d_3d(probs, 1)


blank = np.where(sum(sum(input_data>0)) == 0)
probs[:,:,blank] = 0
probs = np.float16(probs)
np.save('probs_temp.npy',probs)
print 'time elapsed: ' + str(time()-t)
ndp.plot(probs,slice=2)

~~~
# To convert a probability cube to objects using ndparse, one could use the following sequence:

# 3D probs array

import numpy as np
import ndparse as ndp
probs = np.load('probs_temp.npy')
obj = ndp.algorithms.basic_objectify(probs,threshold=0.5, min_size=10,max_size=5000)
ndp.plot(obj,slice=10)

~~~

To plot ndio obtained (RAMON or numpy array) data:

~~~python
import ndparse as ndp
import ndio.remote.neurodata as neurodata
nd = neurodata()
token = 'kasthuri11cc'
channel = 'image'
im2 = nd.get_volume('ac3ac4','ac4_synapse_truth', 4400,5400, 5440, 6440, 1100, 1102, resolution=1)
im = nd.get_volume(token, channel, 4400, 5400, 5440, 6440, 1100, 1102, resolution=1)
ndp.plot(im,im2,slice=1, alpha=0.5)
~~~


ffmpeg library:
https://github.com/imageio/imageio-binaries/tree/master/ffmpeg


import ndio.remote.neurodata as neurodata
nd = neurodata()
import ndio.ramon as ramon
import numpy as np
r = ramon.RAMONSegment()
c = np.round(np.random.rand(5,5)*100,0)
c = np.uint8(c)
r.cutout = c
nd.post_ramon('ndio_demos','ramontests',r)
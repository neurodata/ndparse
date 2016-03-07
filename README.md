**ndparse** is the NeuroData Parse parent repository

ndparse:  NeuroData Parse
=========================

Installation
------------

You can either clone this repository and use it locally, or install from pypi:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pip install ndparse #not yet!
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use this Python library to easily interface with NeuroData algorithms to
manually annotate data, use computer vision algorithms and deploy
across large neuroscience datasets.

This repo contains the code needed to train, evaluate, and deploy code for parsing volumes of NeuroData images. 

It contains the legacy code for manno and macho.  The current version of ndod is divided into the three major components required to parse neuroscience data at scale:

- **mana**: manual annotation protocols and tools
- **maca**: big-data research algorithms to inform neuroscience
- **mad**: machine annotation for deployment and scaling.

To just use one of these, say **mana**, in python you can (and should) type the following: `from ndparse import mana`

*This repo is under extremely active development during the first quarter of 2016.  The previous version of mano, macho and ndod code may be found in [ndod](https://github.com/neurodata/ndod).  The core code that is used for computer vision by the neurodata team will be transitioned to a pip installable python package in the next few weeks.  Stay tuned.*

~~~
pip install conda
conda create -n ndparse2 -c ilastik ilastik-everything-but-tracking
source activate ndparse2
conda install ipython notebook
conda install requests gcc
pip install blosc
pip install ndio
~~~


For ilastik processing:

~~~

~~~

For

except:
    pass

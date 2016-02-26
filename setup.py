from distutils.core import setup
import ndparse

VERSION = ndparse.version

setup(
    name = 'ndparse',
    packages = [
        'ndparse'
        '.annotate',
        #'.algorithms',
        #'.learn',
        #'.deploy'
        #'.utils'

    ],
    version = VERSION,
    description = 'A Python library for NeuroData computer vision and data processing',
    author = 'William Gray Roncal',
    author_email = 'wgr@jhu.edu',
    url = 'https://github.com/neurodata/ndparse',
    download_url = 'https://github.com/neurodata/ndparse/tarball/' + VERSION,
    keywords = [
        'NeuroData',
        'object detection',
        'annotation',
        'computer vision'
    ],
    classifiers = [],
)

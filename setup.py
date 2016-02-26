import ndparse

VERSION = ndparse.version

from distutils.core import setup
setup(
    name='ndparse',
    packages=[
        'ndparse',
        'ndparse.annotate',
        'ndparse.algorithms',
        'ndparse.learn',
        'ndparse.deploy',
        'ndparse.utils'
    ],
    version=VERSION,
    description='A Python library for NeuroData computer vision and data processing',
    author='William Gray Roncal',
    author_email='wgr@jhu.edu',
    url='https://github.com/neurodata/ndparse',
    download_url = 'https://github.com/neurodata/ndparse/tarball/' + VERSION,
    keywords=[
        'NeuroData',
        'object detection',
        'annotation',
        'computer vision'
    ],
    classifiers = [],
)

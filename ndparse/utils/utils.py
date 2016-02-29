import numpy as np


def compute_centroids(data):

    centroids = 5

    return centroids


def concatenate_files(files, outf):
    with open(outf, "a") as fout:
        [fout.write(line) for f in files for line in open(f)]
    pass

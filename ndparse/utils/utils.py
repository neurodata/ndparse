import numpy as np


def compute_centroids(object_matrix, preserve_ids=False, round_val=False):

    # if ids=true, then write a matrix equal to size of maximum
    # value, else, order in object label order

    # if round = true, round centroid coordinates to nearest integer
    # when rounding, TODO: make sure we don't leave the volume

    import skimage.measure as measure

    centroids = []
    # Threshold data
    rp = measure.regionprops(object_matrix)

    for r in rp:
        if round_val > 0:
            centroids.append(np.round(r.Centroid, round_val))
        else:
            centroids.append(r.Centroid)

    return centroids


def concatenate_files(files, outf):
    with open(outf, "a") as fout:
        [fout.write(line) for f in files for line in open(f)]
    pass


def crop_vol(vol, crop_by_dim):

    import skimage.util

    # TODO:  take care of offset
    vol = skimage.util.crop(vol, crop_by_dim)
    return vol


def remove_border_objects(vol):

    from skimage.segmentation import clear_border

    # remove objects intersecting with image border
    vol = clear_border(vol)
    return vol


def relabel_objects(vol):
    # relabel objects starting from 1
    import mahotas
    vol_out, n = mahotas.labeled.relabel(vol)
    return vol_out


def choose_channel_4d_3d(vol, channel):
    """
    Helper function to choose an individual channel from a cube

    Arguments:
        predictions:  RAMONVolume containing a numpy array (x,y,z)

    Returns:
        pixel_out: The raw trained classifier
    """

    prob_channel = vol[:, :, :, channel]

    return prob_channel

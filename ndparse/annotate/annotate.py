from __future__ import absolute_import


class annotate:

    def __init__(self):
        """
        Initialize annotation class.

        Returns:
            None

        """
        pass

    def get_mana_volume(self, token, channel, x_start, x_stop, y_start,
                        y_stop, z_start, z_stop, resolution,
                        server='openconnecto.me', remote='neurodata',
                        outdir='.'):
        """
        Use ndio to get volume from a remote and convert to NIFTI.
        We recommend users use ITK-Snap for annotations.

        Arguments:
            token (str): Token to identify data to download
            channel (str): Channel
            resolution (int): Resolution level
            Q_start (int): The lower bound of dimension 'Q'
            Q_stop (int): The upper bound of dimension 'Q'
            server: default server for remote
            remote: name for remote to use
            outdir: location for nifti file

        Returns:
            Downloaded data.

        """
        import ndio.convert.nifti as ndnifti
        import os.path
        if remote is 'neurodata':
            import ndio.remote.neurodata as ND
            nd = ND(server)
        else:
            raise ValueError("remote option not implemented.")

        image = nd.get_cutout(token, channel, x_start, x_stop, y_start,
                              y_stop, z_start, z_stop, resolution=resolution)

        fileout = '{}_{}_x{}-{}_y{}-{}_z{}-{}_r{}' \
                  '.nii'.format(token, channel, x_start, x_stop,
                                y_start, y_stop, z_start, z_stop,
                                resolution)
        print fileout
        ndnifti.export_nifti(os.path.abspath(os.path.join(
                                             outdir, fileout)), image)

        return fileout

    def put_ramon_volume(self, token, channel, annofile, ramonobj, x_start, x_stop, y_start,
                              y_stop, z_start, z_stop, resolution=1, conncomp=0, remote='neurodata'):
        """
        Use ndio to put annotated nifti volume to a remote
        This first prototype only uploads annotation labels and does no processing

        TODO:  Extend to parse upload params from filename

        Arguments:
            data: nifti volume to convert

        Returns:
            Success or failure
        """

        import ndio.convert.nifti as ndnifti

        if remote is 'neurodata':
            import ndio.remote.neurodata as ND
            nd = ND(remote)
        else:
            raise ValueError("remote option not implemented.")

        anno = ndnifti.import_nifti(annofile)

        if conncomp is 1:
            from skimage.measure import label
            anno, n_label = label(anno, return_num=True)

        return anno
        # relabel ids from 1

#        nd.reserve_ids(token, channel, n_label)



    def put_batch_ramon_meta(self, token, channel, resolution, file, ramonobj):

        pass
        #RAMON
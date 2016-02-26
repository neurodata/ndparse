from __future__ import absolute_import


class mana:

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

        fileout = 'imvol_token_{}_channel_{}_xstart_{}_xstop_{}' \
                  '_ystart_{}_ystop_{}_zstart_{}_zstop_{}_res_{}' \
                  '.nii'.format(token, channel, x_start, x_stop,
                                y_start, y_stop, z_start, z_stop,
                                resolution)
        print fileout
        ndnifti.export_nifti(os.path.abspath(os.path.join(
                                             outdir, fileout)), image)

        return fileout

    def put_mana_volume(self, data):
        """
        Use ndio to put annotated nifti volume to a remote

        Arguments:
            data: nifti volume to convert

        Returns:
            Success or failure
        """
        pass


import numpy as np
import mahotas


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
        print(fileout)
        ndnifti.export_nifti(os.path.abspath(os.path.join(
                                             outdir, fileout)), image)

        return fileout

    def create_ramon_volume(self, token, channel, annofile, ramonobj,
                            conncomp=0, remote='neurodata'):

        """
        Use ndio to put annotated nifti volume to a remote
        This first prototype only uploads annotation labels

        TODO:  Extend to parse upload params from filename

        Arguments:
            data: nifti volume to convert

        Returns:
            Success or failure
        """

        import ndio.convert.nifti as ndnifti

        if remote is 'neurodata':
            import ndio.remote.neurodata as ND
            nd = ND()
        else:
            raise ValueError("remote option not implemented.")

        anno = ndnifti.import_nifti(annofile)
        anno = np.int32(anno)
        if conncomp is 1:
            anno = mahotas.labeled.label(anno, Bc=np.ones([3, 3, 3]))[0]

        # relabel ids from 1
        print('relabeling IDs...')
        anno, n_label = mahotas.labeled.relabel(anno)

        print('reserving IDs...')
        n_label = int(n_label)
        ids = nd.reserve_ids(token, channel, n_label)

        # ids is 0 indexed
        # anno begins at 1
        # TODO: guarantee that these are contiguous

        anno[anno > 0] += ids[0] - 1

        ramon_list = []

        for x in range(0, n_label):
            r = ramonobj
            r.id = ids[x]
            ramon_list.append(r)

        return anno, ramon_list

    def put_ramon_volume(self, token, channel, annofile, ramonobj, x_start,
                         y_start, z_start, resolution=1, conncomp=0,
                         remote='neurodata'):

        vol, ramons = self.create_ramon_volume(token, channel, annofile,
                                               ramonobj, conncomp=conncomp,
                                               remote='neurodata')

        if remote is 'neurodata':
            import ndio.remote.neurodata as neurodata
            nd = neurodata()
        else:
            raise ValueError("remote option not implemented.")

        # upload paint
        print('uploading paint...')
        nd.post_cutout(token, channel,
                       x_start, y_start, z_start,
                       vol, resolution=1)

        # upload ramon
        print('uploading RAMON...')
        print('Sorry, I can\'t upload RAMONObjects yet...' \
              'waiting for new functionality.')

        # for r in ramons:
        #     nd.post_ramon(token, channel, r)

    def put_batch_ramon_meta(self, token, channel, resolution, file, ramonobj):
        # RAMON

        pass

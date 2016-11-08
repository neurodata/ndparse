import numpy as np
import mahotas


def pr_object(detect, truth, overlap=10):

    # we assume that both truth and detect volumes are separate objects

    from scipy import stats

    # TODO:  64-bit support

    # manual relabel (could be slow!)
    utruth = np.unique(truth)
    utruth = utruth[utruth > 0]
    udetect = np.unique(detect)
    udetect = udetect[udetect > 0]

    tp = 0.0
    fp = 0.0
    fn = 0.0

    # TODO:  removing only greatest match

    # for each truth object find a detection

    for t in utruth:  # background is ignored
        match = detect[truth == t]
        match = match[match > 0]  # get rid of spurious values
        match = stats.mode(match)

        if match[1] >= overlap:
            tp += 1

            # any detected objects can only be used once, so remove them here.
            # detect = mahotas.labeled.remove_regions(detect, match[0])
            detect[detect == match[0]] = 0
        else:
            fn += 1

    # detect_left, fp = mahotas.labeled.relabel(detect)
    fp = np.unique(detect)
    fp = fp[fp > 0]
    fp = len(fp)

    precision = 0
    recall = 0

    if tp + fp > 0:
        precision = tp/(tp+fp)

    if tp + fn > 0:
        recall = tp/(tp+fn)

    if (precision == 0) or (recall == 0):
        f1 = 0
    else:
        f1 = (2*precision*recall)/(precision+recall)

    print(precision)
    print(recall)
    print(f1)

    return precision, recall, f1


def find_operating_point(recall_vec, precision_vec, op_point='f1'):

    if op_point is 'f1':
        pass
    else:
        raise('Not implemented')

    pass


def pareto_front(vals1, vals2, round_val=3):

    # butter and guns pareto front.  Removes points not on
    # the pareto frontier

    # round very similar vals
    vals1 = round(vals1, round_val)
    vals2 = round(vals2, round_val)

    v1_out = []
    v2_out = []
    idx_out = []
    for idx in range(0, len(vals1)):

        is_better = np.find(vals1 >= vals1[idx] and vals2 >= vals2[idx])
        if is_better is None:
            v1_out.append(vals1[idx])
            v2_out.append(vals2[idx])
            idx_out.append(idx)

    return v1_out, v2_out, idx_out


def display_pr_curve(precision, recall):
    # following examples from sklearn

    # TODO:  f1 operating point

    import pylab as plt
    # Plot Precision-Recall curve
    plt.clf()
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall example: Max f1={0:0.2f}'.format(max_f1))
    plt.legend(loc="lower left")
    plt.show()


def pr_curve_objectify_sweep(probs, truth, min_sizes=0, max_sizes=1e9,
                             thresholds=np.arange(0.3, 0.03, 1.03)):
    import ndparse.algorithms
    p_min = []
    p_max = []
    p_thresh = []
    p_recall = []
    p_precision = []
    p_f1 = []

    for thresh in thresholds:
        print(thresh)
        for min in min_sizes:
            for max in max_sizes:

                detect = ndparse.algorithms.basic_objectify(probs, thresh,
                                                            min, max)
                p, r, f1 = pr_object(truth, detect)

                p_min.append(min)
                p_max.append(max)
                p_thresh.append(np.round(thresh, 2))
                p_recall.append(r)
                p_precision.append(p)
                p_f1.append(f1)

    return p_recall, p_precision, p_f1, p_min, p_max, p_thresh


def gen_ramon_graph(token_synapse, channel_synapse,
                    token_neurons, channel_neurons,
                    resolution, is_directed=False,
                    save_graphml=None):

    # TODO support segment graphs
    # TODO support filter synapses
    # TODO support multiple servers
    # TODO support other graph output options
    # TODO support enriched graph

    import numpy as np
    import ndio.remote.neurodata as ND
    import ndio.ramon as ramon
    import networkx as nx

    nd = ND()

    id_synapse = nd.get_ramon_ids(token_synapse, channel_synapse,
                                  ramon_type=ramon.RAMONSynapse)
    print('There are: {} synapses'.format(len(id_synapse)))

    # Instantiate graph instance
    if is_directed is False:  # undirected case
        G = nx.Graph()
    else:
        G = nx.DiGraph()

    # for each synapse

    for x in range(np.shape(id_synapse)[0]):
        
        #print(str(x).zfill(4),end=" ")
        
        s = nd.get_ramon_metadata(token_synapse, channel_synapse,
                                  [id_synapse[x]])[0]

        # for each segment
        segments = s.segments[:, 0]
        direction = s.segments[:, 1]  # 1: axon/pre, 2: dendrite/post
        # print direction
        if len(segments) != 2:
            print('multi-way synapses not implemented!')
            raise

        s1 = nd.get_ramon_metadata(token_neurons, channel_neurons,
                                   [segments[0]])[0]
        n1 = s1.neuron
        s2 = nd.get_ramon_metadata(token_neurons, channel_neurons,
                                   [segments[1]])[0]
        n2 = s2.neuron

        if is_directed is False or (direction[0] == 1 and direction[1] == 2):
            if G.has_edge(n1, n2):  # edge already exists, increase weight
                G[n1][n2]['weight'] += 1
            else:
                # new edge. add with weight=1
                G.add_edge(n1, n2, weight=1)

        elif direction[0] == 2 and direction[1] == 1:
            if G.has_edge(n2, n1):  # edge already exists, increase weight
                G[n2][n1]['weight'] += 1
            else:
                # new edge. add with weight=1
                G.add_edge(n2, n1, weight=1)
        else:
            print('1 pre and 1 post synaptic partner are'
                  ' required for directed graph estimation.')
            raise

        if save_file is not None:

            # Save graphml graph
            nx.write_graphml(G, save_graphml)


def plot(im1, im2=None, cmap1='gray', cmap2='jet', slice=0,
         alpha=1, show_plot=True, save_plot=False):
    """
    Convenience function to handle plotting of neurodata arrays.
    Mostly tested with 8-bit image and 32-bit annos, but lots of
    things should work.  Mimics (and uses) matplotlib, but transparently
    deals with RAMON objects, transposes, and overlays.  Slices 3D arrays
    and allows for different blending when using overlays.  We require
    (but verify) that dimensions are the same when overlaying.

    Arguments:
        im1 (array): RAMONObject or numpy array
        im2 (array) [None]:  RAMONObject or numpy array
        cmap1 (string ['gray']): Colormap for base image
        cmap2 (string ['jet']): Colormap for overlay image
        slice (int) [0]: Used to choose slice from 3D array
        alpha (float) [1]: Used to set blending option between 0-1

    Returns:
        None.

    """

    import numpy as np
    import matplotlib.pyplot as plt

    # get im1_proc as 2D array
    fig = plt.figure()
    # fig.set_size_inches(2, 2)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    base_image = False
    im1_proc = None

    if hasattr(im1, 'cutout') and im1.cutout is not None:
        im1_proc = im1.cutout
    elif im1 is not None:
        im1_proc = im1

    if im1_proc is not None and len(np.shape(im1_proc)) == 3:
        im1_proc = im1_proc[:, :, slice]

    if im1_proc is not None:
        base_image = True

    # get im2_proc as 2D array if exists
    overlay_image = False
    im2_proc = None

    if im2 is not None:

        if hasattr(im2, 'cutout') and im2.cutout is not None:
            im2_proc = im2.cutout
        elif im2 is not None:
            im2_proc = im2

        if im2_proc is not None and len(np.shape(im2_proc)) == 3:
            im2_proc = im2_proc[:, :, slice]

    if im2_proc is not None and np.shape(im1_proc) == np.shape(im2_proc):
        overlay_image = True

    if base_image:

        plt.imshow(im1_proc.T, cmap=cmap1, interpolation='bilinear')

    if base_image and overlay_image and alpha == 1:
        # This option is often recommended but seems less good in general.
        # Produces nice solid overlays for things like ground truth
        im2_proc = np.ma.masked_where(im2_proc == 0, im2_proc)
        plt.imshow(im2_proc.T, cmap=cmap2, interpolation='nearest')

    elif base_image and overlay_image and alpha < 1:

        plt.hold(True)
        im2_proc = np.asarray(im2_proc, dtype='float')  # TODO better way
        im2_proc[im2_proc == 0] = np.nan  # zero out bg
        plt.imshow(im2_proc.T, cmap=cmap2,
                   alpha=alpha, interpolation='nearest')

    if save_plot is not False:
        # TODO: White-space
        plt.savefig(save_plot, dpi=300, pad_inches=0)

    if show_plot is True:
        plt.show()

    pass


def save_movie(im1, im2=None, cmap1='gray', cmap2='jet', alpha=1,
               fps=1, outFile='test.mp4'):

    # TODO properly nest plot function

    import moviepy.editor as mpy
    from moviepy.video.io.bindings import mplfig_to_npimage
    import numpy as np
    import matplotlib
    matplotlib.use('TkAgg')

    import matplotlib.pyplot as plt
    time = list(range(0, int(np.shape(im1)[2])))

    fig = plt.figure()
    fig.set_size_inches(10, 10)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    def animate(slice):

        import numpy as np
        import matplotlib.pyplot as plt
        # get im1_proc as 2D array

        base_image = False
        im1_proc = None

        if hasattr(im1, 'cutout') and im1.cutout is not None:
            im1_proc = im1.cutout
        elif im1 is not None:
            im1_proc = im1

        if im1_proc is not None and len(np.shape(im1_proc)) == 3:
            im1_proc = im1_proc[:, :, slice]

        if im1_proc is not None:
            base_image = True

        # get im2_proc as 2D array if exists
        overlay_image = False
        im2_proc = None

        if im2 is not None:

            if hasattr(im2, 'cutout') and im2.cutout is not None:
                im2_proc = im2.cutout
            elif im2 is not None:
                im2_proc = im2

            if im2_proc is not None and len(np.shape(im2_proc)) == 3:
                im2_proc = im2_proc[:, :, slice]

        if im2_proc is not None and np.shape(im1_proc) == np.shape(im2_proc):
            overlay_image = True

        if base_image:

            plt.imshow(im1_proc.T, cmap=cmap1, interpolation='bilinear')

        if base_image and overlay_image and alpha == 1:
            # This option is often recommended but seems less good in general.
            # Produces nice solid overlays for things like ground truth
            im2_proc = np.ma.masked_where(im2_proc == 0, im2_proc)
            plt.imshow(im2_proc.T, cmap=cmap2, interpolation='nearest')

        elif base_image and overlay_image and alpha < 1:

            plt.hold(True)
            im2_proc = np.asarray(im2_proc, dtype='float')  # TODO better way
            im2_proc[im2_proc == 0] = np.nan  # zero out bg
            plt.imshow(im2_proc.T, cmap=cmap2,
                       alpha=alpha, interpolation='nearest')

        return mplfig_to_npimage(fig)

    animation = mpy.VideoClip(animate, duration=len(time))

    import os.path
    extension = os.path.splitext(outFile)[1]

    if extension is 'gif':
        animation.write_gif(outFile, fps=fps, fuzz=0)

    else:  # 'mp4'
        animation.write_videofile(outFile, fps=fps, bitrate='5000k',
                                  codec='libx264')


def print_ramon(ramon):
    """
    This convenience function allows users to see all information contained
    in a RAMON Object.  Initially just the result of vars
    :param ramon: RAMONObject
    :return: variables and fields for object
    """

    vars(ramon)

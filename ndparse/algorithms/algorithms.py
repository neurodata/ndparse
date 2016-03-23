from __future__ import absolute_import
import os
import numpy as np
import six


def basic_objectify(predictions, threshold, min_size, max_size, relabel=True):

    import scipy.ndimage.measurements

    # Threshold data
    obj = predictions > threshold
    object = np.zeros_like(obj, dtype = 'uint32')
    # Data properties

    label, n = scipy.ndimage.measurements.label(obj, return_num=True)
    rp = measure.regionprops(label)

    print 'Identified {} raw labels.'.format(n)

    # Remove size outliers
    new_idx = 1

    for r in rp:
        if r.Area >= min_size and r.Area <= max_size:  # keep object

            if relabel:
                object[obj == r.label] = new_idx
                new_idx += 1
            else:
                object[obj == r.label] = r.label

        return object


def run_ilastik_pixel(input_data, classifier, threads=2, ram=2000):

    """
    Runs a pre-trained ilastik classifier on a volume of data
    Adapted from Stuart Berg's example here: https://github.com/ilastik/ilastik/blob/master/examples/example_python_client.py

    Arguments:
        input_data:  RAMONVolume containing a numpy array or raw numpy array (x,y,z)

    Returns:
        pixel_out: The raw trained classifier
    """

    from collections import OrderedDict
    import vigra
    import os
    import ilastik_main
    from ilastik.applets.dataSelection import DatasetInfo
    from ilastik.workflows.pixelClassification import PixelClassificationWorkflow

    # Before we start ilastik, optionally prepare these environment variable settings.
    os.environ["LAZYFLOW_THREADS"] = str(threads)
    os.environ["LAZYFLOW_TOTAL_RAM_MB"] = str(ram)

    # Programmatically set the command-line arguments directly into the argparse.Namespace object
    # Provide your project file, and don't forget to specify headless.
    args = ilastik_main.parser.parse_args([])
    args.headless = True
    args.project = classifier

    # Instantiate the 'shell', (in this case, an instance of ilastik.shell.HeadlessShell)
    # This also loads the project file into shell.projectManager
    shell = ilastik_main.main(args)
    assert isinstance(shell.workflow, PixelClassificationWorkflow)

    # Obtain the training operator
    opPixelClassification = shell.workflow.pcApplet.topLevelOperator

    # Sanity checks
    assert len(opPixelClassification.InputImages) > 0
    assert opPixelClassification.Classifier.ready()

    # For this example, we'll use random input data to "batch process"
    print input_data.shape

    # In this example, we're using 2D data (with an extra dimension for  channel).
    # Tagging the data this way ensures that ilastik interprets the axes correctly.
    input_data = vigra.taggedView(input_data, 'xyz')

    # In case you're curious about which label class is which,
    # let's read the label names from the project file.
    label_names = opPixelClassification.LabelNames.value
    label_colors = opPixelClassification.LabelColors.value
    probability_colors = opPixelClassification.PmapColors.value

    print label_names, label_colors, probability_colors

    # Construct an OrderedDict of role-names -> DatasetInfos
    # (See PixelClassificationWorkflow.ROLE_NAMES)
    role_data_dict = OrderedDict([("Raw Data", [DatasetInfo(preloaded_array=input_data)])])

    # Run the export via the BatchProcessingApplet
    # Note: If you don't provide export_to_array, then the results will
    #       be exported to disk according to your project's DataExport settings.
    #       In that case, run_export() returns None.
    predictions = shell.workflow.batchProcessingApplet.run_export(role_data_dict, export_to_array=True)
    predictions = np.squeeze(predictions)
    print predictions.dtype, predictions.shape

    print "DONE."

    return predictions

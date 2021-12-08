"""General helper functions."""

import numpy as np
import os
import datetime
import cPickle as pickle
import pandas as pd

import matplotlib.pyplot as plt
from warnings import warn
from xml.etree import ElementTree
from distutils.version import LooseVersion

import cv2
cv2.setNumThreads(0)
from PIL import Image, ImageEnhance


def savefigs(pdf_pages, figs):
    """Save a single figure or list of figures to a multi-page PDF.

    This function is mostly used so that the same call can be used for a single
    page or multiple pages. Will close Figures once they are written.

    Parameters
    ----------
    pdf_pages : matplotlib.backends.backend_pdf.PdfPages
        PdfPage instance that the figures will get written to.
    figs : matplotlib.pyplot.Figure or iterable of Figures

    """
    try:
        for fig in figs:
            pdf_pages.savefig(fig)
            plt.close(fig)
    except TypeError:
        pdf_pages.savefig(figs)
        plt.close(figs)


def save_figure(
        fig, filename, save_dir='', expt_grps=None, stats_data=None,
        ignore_shuffle=True):
    """Helper function to save figure and run stats.

    Parameters
    ----------
    fig : matplotlib.pyplot.Figure
    filename : str
    save_dir : str
    expt_grps : optional, sequence of lab.ExperimentGroup
        If passed in and saving as a pdf, add a page of summary information
        about the experiments used in the analysis.
    stats_data : optional, dict
        If passed in, save data with save_data and also create stat figures
        for all data if writing a pdf. See save_data for details of format.
    ignore_shuffle : bool
        If True, ignore the shuffle data for the ANOVA in any stats figures.

    """
    if not os.path.isdir(os.path.normpath(save_dir)):
        os.makedirs(os.path.normpath(save_dir))

    if filename.endswith('pdf'):
        from matplotlib.backends.backend_pdf import PdfPages
        pp = PdfPages(os.path.join(save_dir, filename))
        if expt_grps is not None:
            savefigs(pp, summarySheet(expt_grps))
        if stats_data is not None:
            # Create stats summary figures for whatever we can.
            for key in stats_data:
                try:
                    savefigs(pp, stats_fig(
                        stats_data[key], label=key,
                        ignore_shuffle=ignore_shuffle))
                except:
                    pass
        savefigs(pp, fig)
        pp.close()
    elif filename.endswith('svg'):
        fig.savefig(os.path.join(
            save_dir, filename.replace('.pdf', '.svg')), format='svg')
    else:
        # If we don't recognize the file format, drop into a debugger
        warn('Unrecognized file format, dropping into debugger.')
        from pudb import set_trace
        set_trace()




def parseTime(timeStr):
    """Parses a time string from the xml into a datetime object"""
    # Check for sql format
    if ':' in timeStr:
        return datetime.datetime.strptime(timeStr, '%Y-%m-%d %H:%M:%S')
    else:
        return datetime.datetime.strptime(timeStr, '%Y-%m-%d-%Hh%Mm%Ss')


def timestamp():
    """Returns the current time as a timestamp string."""
    return datetime.datetime.strftime(
        datetime.datetime.now(), '%Y-%m-%d-%Hh%Mm%Ss')


def norm_to_complex(arr):
    return angle_to_complex(norm_to_angle(arr))


def norm_to_angle(arr):
    return (arr * 2. * np.pi) % (2 * np.pi)


def complex_to_norm(arr):
    return angle_to_norm(complex_to_angle(arr)) % 1.


def complex_to_angle(arr):
    return np.angle(arr) % (2 * np.pi)


def angle_to_norm(arr):
    return (arr / 2. / np.pi) % 1.


def angle_to_complex(arr):
    return np.array([
        np.complex(x, y) for x, y in zip(np.cos(arr), np.sin(arr))])

def dir_start_time(path):
    """Extract a start time from a Prairie xml file."""
    
    for _, elem in ElementTree.iterparse(path, events=("start",)):
        if elem.tag == 'PVScan':
            date_string = elem.get('date')
            time = datetime.datetime.strptime(
                date_string, '%m/%d/%Y %I:%M:%S %p')
        elif elem.tag == 'Frame':
            first_frame_delay = datetime.timedelta(seconds=float(elem.get(
                'absoluteTime')))
            break
    # Make sure we found a first_frame_delay
    first_frame_delay
    # If the parse fails, return a meaningless date, nothing should match
    # this
    return time + first_frame_delay

def unique_rows(array_like):
    """
    Returns array with only unique rows, as well as indices
    of first instance of each row
    """

    seen = set()
    result = []
    idx = []

    tuple_list = [tuple(row) for row in array_like]
    for i, row in enumerate(tuple_list):
        if row not in seen:
            seen.add(row)
            result.append(row)
            idx.append(i)

    return result, idx

def parallel_sort(a, b, key=None):
    """
    Sorts list a according to given key, and then applies
    the same sort to the paired list b
    """

    if key is None:
        new_key = None
    else:
        new_key = lambda x: key(x[0])

    a, b = zip(*sorted(zip(a, b), key=new_key))

    return list(a), list(b)

def get_prairieview_version(xml_filepath):
    """Return Prairieview version number"""
    for _, elem in ElementTree.iterparse(xml_filepath, events=("start",)):
        if elem.tag == 'PVScan':
            return LooseVersion(elem.get('version'))


def get_element_size_um(xml_filepath, prairie_version=None):
    """Determine the size in um of x and y in order to store it with the
    data. The HDF5 plugin for ImageJ will read this metadata"""
    if not prairie_version:
        prairie_version = get_prairieview_version(xml_filepath)

    if prairie_version >= LooseVersion('5.2'):
        for _, elem in ElementTree.iterparse(xml_filepath):
            if elem.get('key') == 'micronsPerPixel':
                for value in elem.findall('IndexedValue'):
                    if value.get('index') == "XAxis":
                        x = float(value.get('value'))
                    elif value.get('index') == "YAxis":
                        y = float(value.get('value'))
                return (1, y, x)
    else:
        for _, elem in ElementTree.iterparse(xml_filepath):
            if elem.tag == 'PVStateShard':
                for key in elem.findall('Key'):
                    if key.get('key') == 'micronsPerPixel_XAxis':
                        x = float(key.get('value'))
                    elif key.get('key') == 'micronsPerPixel_YAxis':
                        y = float(key.get('value'))
                return (1, y, x)
    print('Unable to identify element size, returning default value')
    return (1, 1, 1)


def maxmin_filter(signal, window=300, sigma=5):
    """Calculate baseline as the rolling maximum of the rolling minimum of the
    smoothed trace

    Parameters
    ----------
    signal : array, size (n_ROIs, n_timepoints)
    window : int
        Optional, size of the rolling window for max/min/smoothing
    sigma : int
        Standard deviation of the gaussian smoothing kernel
    """

    kwargs = {'window': window, 'min_periods': int(window / 5),
              'center': True, 'axis': 1}

    smooth_signal = pd.DataFrame(signal).rolling(
        win_type='gaussian', **kwargs).mean(std=sigma)

    return smooth_signal.rolling(**kwargs).min().rolling(**kwargs).max().values


def calc_cdf(ax, values, bins='exact', range=None, **kwargs):
    """Plot the empirical CDF.

    Parameters
    ----------
    ax : matplotlib.pyplot.Axes
        The axis to plot on
    Values : array-like
       The data to be plotted.
    bins
        See matplotlib.pyplot.hist documentation.
        Can also be 'exact' to calculate the exact empirical CDF
    range
        See matplotlib.pyplot.hist documentation.
    **kwargs
        Any additional keyword arguments are passed to the plotting function.

    """
    if bins == 'exact':
        bins = np.unique(np.sort(values))
        if len(bins) == 1:
            return None, None
    hist_counts, hist_bins = np.histogram(values, bins=bins, range=range)

    cum_counts = np.cumsum(hist_counts)
    cdf = cum_counts * 1.0 / cum_counts[-1]

    # Want to plot each value at the right side of the bin, but then also put
    # back in value for the beginning of the first bin
    cdf_zero = np.sum(values <= hist_bins[0]) * 1.0 / cum_counts[-1]
    cdf = np.hstack([cdf_zero, cdf])

    ax.plot(hist_bins, cdf, **kwargs)

    ax.set_ylim((0, 1))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='x', direction='out')
    ax.tick_params(axis='y', direction='out')
    ax.set_ylabel('Cumulative probability')

    return hist_bins, cdf


# Image Preprocessing
def float_to_uint(im):
    im -= np.min(im)
    im /= np.max(im)
    im *= 255
    return im.astype('uint8')


def clip_percentiles(im, percentiles=[0.5, 99.5]):
    percs = np.percentile(im, percentiles)
    im[im < percs[0]] = percs[0]
    im[im > percs[1]] = percs[1]
    return im


def preprocess_image(im, percentiles=[0.5, 99.5]):
    im = float_to_uint(clip_percentiles(im, percentiles))
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(16, 16))
    im = clahe.apply(im)
    im = Image.fromarray(im)
    enhancerCont = ImageEnhance.Sharpness(im)
    im = enhancerCont.enhance(3)
    return np.array(im)

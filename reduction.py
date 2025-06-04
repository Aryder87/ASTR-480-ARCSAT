#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Filename: reduction.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from astropy.io import fits
from astropy.visualization import ImageNormalize, LinearStretch, ZScaleInterval
from astropy.stats import sigma_clip
from astroscrappy import detect_cosmics
import numpy as np 
from matplotlib import pyplot as plt
from photutils.aperture import CircularAperture, aperture_photometry
import glob
import pathlib


def run_reduction(data_dir):
    """This function must run the entire CCD reduction process. You can implement it
    in any way that you want but it must perform a valid reduction for the two
    science frames in the dataset using the functions that you have implemented in
    this module. Then perform aperture photometry on at least one of the science
    frames, using apertures and sky annuli that make sense for the data.

    No specific output is required but make sure the function prints/saves all the
    relevant information to the screen or to a file, and that any plots are saved to
    PNG or PDF files.

    """

    from bias import create_median_bias
    from darks import create_median_dark
    from flats import create_median_flat
    from science import reduce_science_frame
    from ptc import calculate_gain, calculate_readout_noise
    from diff_photometry import differential_photometry, plot_light_curves

    science_list = sorted(pathlib.Path(data_dir).glob('LPSEB*_reprojected.fits'))
    dark_list = sorted(pathlib.Path(data_dir).glob('Dark*.fits'))
    bias_list = sorted(pathlib.Path(data_dir).glob('Bias*.fits'))
    flat_list = sorted(pathlib.Path(data_dir).glob('dome*.fits'))

    median_bias_filename = 'median_bias.fits'
    median_flat_filename = 'normalized_flat.fits'
    median_dark_filename = 'median_dark.fits'
    
    bias = create_median_bias(bias_list, median_bias_filename)
    dark = create_median_dark(dark_list, median_bias_filename, median_dark_filename)
    flat = create_median_flat(flat_list, median_bias_filename, median_flat_filename, median_dark_filename)
    

    science = []
    for i in range(len(science_list)):
        output_file=f"reduced_science{i+1}.fits"
        sci_image = reduce_science_frame(science_list[i],
                                         median_bias_filename,
                                         median_dark_filename,
                                         median_flat_filename,
                                         reduced_science_filename=output_file)
        science.append(output_file)

    #Differential Photometry values LPSEB35	240.184(deg)	+43:08(deg)
    target_radec = (240.184, 43.145)

    #Comparison stars ra and dec
    comp_radec = (
        (240.225, 43.1155),
        (240.2157, 43.1419),
        (240.1297, 43.2169)
    )

    #define image_list and call on our reduced images
    image_list = science

    #call on time observed
    times, diff_flux, comp_fluxes = differential_photometry(image_list, target_radec, comp_radec)

    #plot light curves
    plot_light_curves(times, diff_flux, output="lightcurve.png")

    #calculate gain    
    gain = calculate_gain(flat_list)
    print(f"Gain = {gain:.2f} e-/ADU")

    #calculate readoutnoise
    readout_noise = calculate_readout_noise(bias_list, gain)
    print(f"Readout Noise = {readout_noise:.2f} e-")
    
    return

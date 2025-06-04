#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Filename: bias.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from astropy.io import fits
from astropy.visualization import ImageNormalize, LinearStretch, ZScaleInterval
from astropy.stats import sigma_clip
from astroscrappy import detect_cosmics
import numpy as np 
from matplotlib import pyplot as plt
from photutils.aperture import CircularAperture, aperture_photometry
import glob

def create_median_bias(bias_list, median_bias_filename):

    biases = [] 

    for bias_file in (bias_list):
        bias_data = fits.getdata(bias_file)
        biases.append(bias_data.astype('f4'))

    
    bias_images_masked = sigma_clip(biases, cenfunc='median', sigma=3, axis=0)
    median_bias = np.ma.mean(bias_images_masked, axis=0).data

    bias_hdu = fits.PrimaryHDU(data=median_bias)
    hdul = fits.HDUList([bias_hdu])
    hdul.writeto(median_bias_filename, overwrite=True)

    return median_bias

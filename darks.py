#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Filename: darks.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from astropy.io import fits
from astropy.visualization import ImageNormalize, LinearStretch, ZScaleInterval
from astropy.stats import sigma_clip
from astroscrappy import detect_cosmics
import numpy as np 
from matplotlib import pyplot as plt
from photutils.aperture import CircularAperture, aperture_photometry
import glob

def create_median_dark(dark_list, bias_filename, median_dark_filename):

    dark_bias_data = []
    bias_data = fits.getdata(bias_filename).astype('f4')
    
    for dark_file in (dark_list): 
        darks = fits.open(dark_file)
        dark_data = darks[0].data.astype('f4')
        exptime = darks[0].header['EXPTIME']
        
        dark_minus_bias = dark_data - bias_data
        dark_bias_data.append(dark_minus_bias / exptime)

    dark_sc = sigma_clip(dark_bias_data, cenfunc='median', sigma=2.5, axis=0)

    median_dark = np.ma.mean(dark_sc, axis=0).data


    dark_hdu = fits.PrimaryHDU(data=median_dark, header=darks[0].header)
    dark_hdu.header['EXPTIME'] = 1
    dark_hdu.header['COMMENT'] = 'Combiend dark image'
    dark_hdu.header['BIASFILE'] = ('median_bias.fits', 'Bias image used to subtract from darks')
   
    dark_hdu.writeto(median_dark_filename, overwrite=True)    

    return median_dark
    
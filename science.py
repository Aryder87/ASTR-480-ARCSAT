#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Filename: science.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from astropy.io import fits
from astropy.visualization import ImageNormalize, LinearStretch, ZScaleInterval
from astropy.stats import sigma_clip
from astroscrappy import detect_cosmics
import numpy as np 
from matplotlib import pyplot as plt
from photutils.aperture import CircularAperture, aperture_photometry
import glob

def reduce_science_frame(

    science_filename,
    median_bias_filename,
    median_dark_filename,
    median_flat_filename,
    reduced_science_filename="reduced_science.fits",
):

    
    science = fits.open(science_filename)
    science_image = fits.getdata(science_filename).astype('f4')
    bias = fits.getdata(median_bias_filename).astype('f4')
    flat = fits.getdata(median_flat_filename).astype('f4')
    dark = fits.getdata(median_dark_filename).astype('f4')
    exptime = fits.getheader(science_filename).get('EXPTIME')

    reduced_science = (science_image - bias - (dark * exptime))/(flat)
    reduced_science = reduced_science[100:-100, 100:-100]
    mask, cleaned = detect_cosmics(reduced_science)
    reduced_science = cleaned

    norm_orig = ImageNormalize(science_image[100:-100, 100:-100], interval=ZScaleInterval(), stretch=LinearStretch())
    norm = ImageNormalize(reduced_science, interval=ZScaleInterval(), stretch=LinearStretch())

    fig, axes = plt.subplots(1, 2, figsize=(8, 12))
    _ = axes[0].imshow(science_image[100:-100, 100:-100], origin='lower', norm=norm_orig, cmap='YlOrBr_r')
    plt.close()
    _ = axes[1].imshow(reduced_science, origin='lower', norm=norm, cmap='YlOrBr_r')
    plt.close()

    science_hdu = fits.PrimaryHDU(data=reduced_science, header=science[0].header)
    science_hdu.header['COMMENT'] = 'Final science image'
    science_hdu.header['BIASFILE'] = ('bias.fits', 'Bias image used to subtract bias level')
    science_hdu.header['DARKFILE'] = ('dark.fits', 'Dark image used to subtract dark current')
    science_hdu.header['FLATFILE'] = ('flat_g.fits', 'Flat-field image used to correct flat-fielding')
    science_hdu.writeto(reduced_science_filename, overwrite=True) 

    return reduced_science
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Filename: science.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from astropy.io import fits
from astropy.visualization import ImageNormalize, LinearStretch, ZScaleInterval
from astroscrappy import detect_cosmics
import numpy as np 
from matplotlib import pyplot as plt
import glob
import gc

def reduce_science_frame(
    science_filename,
    median_bias_filename,
    median_dark_filename,
    median_flat_filename,
    reduced_science_filename
):

    
    with fits.open(science_filename) as science:
        science_image = science[0].data.astype('f4')
        header = science[0].header.copy()
        exptime = fits.getheader(science_filename).get('EXPTIME')
    bias = fits.getdata(median_bias_filename).astype('f4')
    flat = fits.getdata(median_flat_filename).astype('f4')
    dark = fits.getdata(median_dark_filename).astype('f4')
    

    reduced_science = (science_image - bias - (dark * exptime))/(flat)

    reduced_science = reduced_science[100:-100, 100:-100]

    #Need to crop the edges of the reduced science wcs due to cropping the reduced science image for masking 
    if 'CRPIX1' in header and 'CRPIX2' in header:
        header['CRPIX1'] -= 100
        header['CRPIX2'] -= 100
    header['NAXIS1'] = reduced_science.shape[1]
    header['NAXIS2'] = reduced_science.shape[0]

    _ , cleaned = detect_cosmics(reduced_science)

    norm_orig = ImageNormalize(science_image, interval=ZScaleInterval(), stretch=LinearStretch())
    norm = ImageNormalize(cleaned, interval=ZScaleInterval(), stretch=LinearStretch())

    fig, axes = plt.subplots(1, 2, figsize=(8, 12))
    axes[0].imshow(science_image, origin='lower', norm=norm_orig, cmap='YlOrBr_r')
    axes[1].imshow(cleaned, origin='lower', norm=norm, cmap='YlOrBr_r')
    #plt.savefig(f"{reduced_science_filename}_compare.png", dpi=300)
    plt.close()

    science_hdu = fits.PrimaryHDU(data=cleaned, header=science[0].header)
    science_hdu.header['COMMENT'] = 'Final science image'
    science_hdu.header['BIASFILE'] = ('bias.fits', 'Bias image used to subtract bias level')
    science_hdu.header['DARKFILE'] = ('dark.fits', 'Dark image used to subtract dark current')
    science_hdu.header['FLATFILE'] = ('flat_g.fits', 'Flat-field image used to correct flat-fielding')
    science_hdu.writeto(reduced_science_filename, overwrite=True) 

    del science_image, bias, flat, dark, reduced_science, cleaned
    gc.collect()

    return None
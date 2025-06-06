#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Filename: flats.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from astropy.io import fits
from astropy.visualization import ImageNormalize, LinearStretch, ZScaleInterval
from astropy.stats import sigma_clip
from astroscrappy import detect_cosmics
import numpy as np 
from matplotlib import pyplot as plt
from photutils.aperture import CircularAperture, aperture_photometry
import glob

def create_median_flat(flat_list, bias_filename, median_flat_filename, dark_filename=None):

    
    bias_data = fits.getdata(bias_filename).astype('f4')
    processed_flats = []

    first_filter = None
    for flat_file in (flat_list):
        with fits.open(flat_file) as hdu:
            flat_data = hdu[0].data.astype('f4')
            header = hdu[0].header

            current_filter = header.get('FILTER', 'Unknown')
            if first_filter is None:
                first_filter = current_filter
            elif current_filter != first_filter:
                raise ValueError(f"filtter mismatch: {current_filter} != {first_filter}")

            flat_corrected = flat_data - bias_data

            if dark_filename is not None:
                dark_data = fits.getdata(dark_filename).astype('f4')
                flat_exptime = header.get('EXPTIME', 1.0)
                dark_exptime = fits.getheader(dark_filename).get('EXPTIME', 1.0)

                if dark_exptime !=0:
                    scale_factor = flat_exptime / dark_exptime
                    flat_corrected -= dark_data * scale_factor
                else:
                    raise ValueError("Dark frame exposure time is zero or null")

            processed_flats.append(flat_corrected)

    flat_array_3d = np.array(processed_flats)
    clipped_flats = sigma_clip(flat_array_3d, cenfunc='median', sigma=3, axis = 0)
    median_flat = np.ma.median(clipped_flats, axis=0)
    flat_median_value = np.ma.median(median_flat)
    
    if flat_median_value == 0: 
        raise ValueError("Median flat value is zero or null, can't normalize")
        
    normalized_flat = median_flat.data / flat_median_value

    hdu = fits.PrimaryHDU(normalized_flat)
    hdu.header['COMMENT'] = 'Normalized median flat frame'
    hdu.header['Filter'] = first_filter
    hdu.writeto(median_flat_filename, overwrite=True)

    return normalized_flat

def plot_flat(median_flat_filename,
    ouput_filename="median_flat.png",
    profile_ouput_filename="median_flat_profile.png"):

    flat_data = fits.getdata(median_flat_filename).astype('f4')
    
    plt.figure()
    vmin = np.percentile(flat_data, 5)
    vmax = np.percentile(flat_data, 95)
    _ = plt.imshow(flat_data, origin = 'lower', cmap='YlOrBr_r', vmin=vmin, vmax=vmax)
    plt.savefig(f'{ouput_filename}', dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close()

    flat_1d = np.ma.median(flat_data, axis = 0)

    _ = plt.plot(flat_1d)
    plt.savefig(f'{profile_ouput_filename}', dpi=300, bbox_inches='tight')
    plt.close()

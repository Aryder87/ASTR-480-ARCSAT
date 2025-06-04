#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Filename: ptc.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from astropy.io import fits
from astropy.visualization import ImageNormalize, LinearStretch, ZScaleInterval
from astropy.stats import sigma_clip
from astroscrappy import detect_cosmics
import numpy as np 
from matplotlib import pyplot as plt
from photutils.aperture import CircularAperture, aperture_photometry
import glob

def calculate_gain(files):
    """This function must:

    - Accept a list of files that you need to calculate the gain
      (two files should be enough, but what kind?).
    - Read the files and calculate the gain in e-/ADU.
    - Return the gain in e-/ADU.

    """
    flats = []

    for flat in (files):
        header = fits.getheader(flat)
        image_type = header['IMAGETYP']

        if image_type != 'FLAT':
            raise Exception(f'Flats calculate gain') 

        data = fits.getdata(flat)
        flats.append(data.astype('f4'))
                     

    flat_diff = flats[1] - flats[0]
    flat_diff_var = np.var(flat_diff)

    mean_signal = 0.5 * np.mean(flats[0] + flats[1])

    gain = 2 * mean_signal / flat_diff_var

    print(f'Gain: {gain:.2f} e-/ADU')

    return float(gain)


def calculate_readout_noise(files, gain):
    """This function must:

    - Accept a list of files that you need to calculate the readout noise
      (two files should be enough, but what kind?).
    - Accept the gain in e-/ADU as gain. This should be the one you calculated
      in calculate_gain.
    - Read the files and calculate the readout noise in e-.
    - Return the readout noise in e-.

    """
    biases = []

    for bias in (files):
        header = fits.getheader(bias)
        image_type = header['IMAGETYP']

        if image_type != 'BIAS':
            raise Exception(f'Bias calculates readout noise') 

        data = fits.getdata(bias)
        biases.append(data.astype('f4'))
    

    bias_diff = biases[1] - biases[0]
    bias_diff_var = np.var(bias_diff)

    readout_noise_adu = np.sqrt(bias_diff_var / 2)
    readout_noise = readout_noise_adu * gain

    print(f'Readout noise (ADU): {readout_noise_adu:.2f} ADU')
    print(f'Readout noise (e-): {readout_noise:.2f} e-')
    
    return float(readout_noise)

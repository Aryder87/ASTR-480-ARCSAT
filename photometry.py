#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Filename: photometry.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from astropy.io import fits
from astropy.visualization import ImageNormalize, LinearStretch, ZScaleInterval
from astropy.stats import sigma_clip
from astroscrappy import detect_cosmics
import numpy as np
from matplotlib import pyplot as plt
from photutils.aperture import CircularAperture, aperture_photometry, CircularAnnulus
import glob
from astropy.table import Table, vstack

def do_aperture_photometry(
    image,
    positions,
    radii,
    sky_radius_in,
    sky_annulus_width,
):
    """
    Perform aperture photometry on a science image with sky background subtraction.

    Parameters:
    image (str or ndarray): Path to a FITS file or a 2D NumPy array of the science image.
    positions (list): List of (x, y) tuples for source positions.
    radii (list or float or array): List of aperture radii (in pixels) for photometry,
        or a single radius.
    sky_radius_in (float): Inner radius of the sky annulus (in pixels).
    sky_annulus_width (float): Width of the sky annulus (in pixels).

    Returns:
    astropy.table.Table: Table containing photometry results for each position and radius.
    """
    # Load image if it's a file path
    if isinstance(image, str):
        try:
            with fits.open(image) as hdul:
                data = hdul[0].data
                if not isinstance(data, np.ndarray) or data.ndim != 2:
                    raise ValueError("Image must be a 2D array.")
        except Exception as e:
            raise ValueError(f"Failed to load FITS file {image}: {str(e)}")
    else:
        data = image
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise ValueError("Image must be a 2D array.")

    # Validate inputs
    if not positions or not all(isinstance(pos, tuple) and len(pos) == 2 for pos in positions):
        raise ValueError("Positions must be a list of (x, y) tuples.")

    # Convert radii to a list if it's a single value or array
    if isinstance(radii, (int, float)):
        radii = [float(radii)]
    elif isinstance(radii, np.ndarray):
        radii = radii.astype(float).tolist()
    if not radii or not all(isinstance(r, (int, float)) and r > 0 for r in radii):
        raise ValueError("Radii must be a list of positive numbers.")

    if not isinstance(sky_radius_in, (int, float)) or sky_radius_in <= 0:
        raise ValueError("sky_radius_in must be a positive number.")
    if not isinstance(sky_annulus_width, (int, float)) or sky_annulus_width <= 0:
        raise ValueError("sky_annulus_width must be a positive number.")

    # Define sky annulus (pixel-based)
    sky_annulus = CircularAnnulus(
        positions=positions,
        r_in=sky_radius_in,
        r_out=sky_radius_in + sky_annulus_width
    )

    # Perform sky background estimation
    try:
        sky_ap_table = aperture_photometry(data, sky_annulus)
        sky_area = sky_ap_table['aperture_sum'] / sky_annulus.area
        sky_background = sky_area  # Mean background per pixel
    except Exception as e:
        raise ValueError(f"Failed to perform sky background photometry: {str(e)}")

    # Perform aperture photometry for each radius
    results = []
    for radius in radii:
        apertures = CircularAperture(positions, r=radius)
        try:
            phot_table = aperture_photometry(data, apertures)
            phot_table['aperture_sum_sky_sub'] = phot_table['aperture_sum'] - sky_background * apertures.area
            phot_table['radius'] = radius
            phot_table.meta.clear()  # Clear metadata to avoid merge conflicts
            results.append(phot_table)
        except Exception as e:
            raise ValueError(f"Failed to perform aperture photometry for radius {radius}: {str(e)}")

    # Combine results into a single table
    try:
        final_table = vstack(results)
    except Exception as e:
        raise ValueError(f"Failed to stack photometry tables: {str(e)}")

    # Add metadata
    final_table.meta['sky_inner_radius'] = sky_radius_in
    final_table.meta['sky_annulus_width'] = sky_annulus_width

    return final_table

def plot_radial_profile(aperture_photometry_data, output_filename="radial_profile.png"):
    """
    Plot the radial profile of one or more targets from aperture photometry data.

    Parameters:
    aperture_photometry_data (astropy.table.Table): Table containing photometry results,
        with columns 'aperture_sum_sky_sub', 'radius', 'xcenter', 'ycenter', and
        metadata 'sky_inner_radius'.
    output_filename (str): Filename to save the plot (default: 'radial_profile.png').

    Returns:
    astropy.table.Table: The input photometry table.
    """
    if not isinstance(aperture_photometry_data, Table):
        raise ValueError("aperture_photometry_data must be an Astropy Table.")
    if 'aperture_sum_sky_sub' not in aperture_photometry_data.colnames or \
       'radius' not in aperture_photometry_data.colnames:
        raise ValueError("Table must contain 'aperture_sum_sky_sub' and 'radius' columns.")
    if 'sky_inner_radius' not in aperture_photometry_data.meta:
        raise ValueError("Table metadata must include 'sky_inner_radius'.")

    sky_inner_radius = aperture_photometry_data.meta['sky_inner_radius']
    try:
        positions = list(set((row['xcenter'], row['ycenter']) 
                            for row in aperture_photometry_data))
    except KeyError as e:
        raise ValueError("Table must contain 'xcenter' and 'ycenter' columns.")

    plt.figure(figsize=(8, 6))
    for i, (x, y) in enumerate(positions):
        mask = (aperture_photometry_data['xcenter'] == x) & \
               (aperture_photometry_data['ycenter'] == y)
        target_data = aperture_photometry_data[mask]
        sorted_indices = np.argsort(target_data['radius'])
        radii = target_data['radius'][sorted_indices]
        fluxes = target_data['aperture_sum_sky_sub'][sorted_indices]
        label = f"Target at ({x:.0f}, {y:.0f})"
        plt.plot(radii, fluxes, marker='o', linestyle='-', label=label)

    plt.axvline(x=sky_inner_radius, color='red', linestyle='--', 
                label=f'Sky annulus inner radius ({sky_inner_radius:.1f} pix)')
    plt.xlabel('Aperture Radius (pixels)')
    plt.ylabel('Sky-Subtracted Flux (ADU)')
    plt.title('Radial Profile of Targets')
    plt.grid(True)
    plt.legend()
    try:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Radial profile plot saved to {output_filename}")
    except Exception as e:
        raise ValueError(f"Failed to save radial profile plot to {output_filename}: {str(e)}")

    return aperture_photometry_data
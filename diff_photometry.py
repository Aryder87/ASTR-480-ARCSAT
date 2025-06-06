import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.visualization import ZScaleInterval, ImageNormalize, LinearStretch
from astropy.table import Table, vstack
from astropy.stats import sigma_clip
from astroscrappy import detect_cosmics
from photutils.aperture import CircularAperture, aperture_photometry, CircularAnnulus
from photutils.centroids import centroid_com
import glob

#Defines a function to calculate centroids given ra and dec using wcs and SkyCoord
def find_centroid_radec(image_file, ra, dec):
    with fits.open(image_file) as hdul:
        data = hdul[0].data
        wcs = WCS(hdul[0].header)
    skycoord = SkyCoord(ra, dec, unit='deg')
    x, y = wcs.world_to_pixel(skycoord)
    if not (0 <= int(x)-10 < data.shape[1] and 0 <= int(y)-10 < data.shape[0]): #ensures our centroid is in bound
        print(f"Initial centroid out of bounds for {image_file}: x={x}, y={y}")
        return None
    
    cutout = data[int(y)-10:int(y)+10, int(x)-10:int(x)+10] #accounts for an area around each coordinate (15x15 cutout)
    try:
        clean_data = sigma_clip(cutout, sigma=3, maxiters=3, masked=False)
        if len(clean_data) != 2:
            print(f"Invalid centroid result for{image_file}: {clean_data}")
            return None
        dx, dy = centroid_com(clean_data)
    except ValueError as e:
        print(f"Centroid computation failed for {image_file}: {e}")
        return None
    max_shift = 10  # Maximum allowed centroid shift in pixels
    if abs(dx) > max_shift or abs(dy) > max_shift:
        print(f"Centroid shift too large for {image_file}: dx={dx}, dy={dy}")
        return None
    return(x - 10 + dx, y - 10 + dy)

#Define a function which generates aperture and annuli using photutils given a radius, sky_rin/rout, and
#Calculates the raw, sky, and background flux of our binary star system
#sky_rin = 17.9 sky_rout = 22, sky_width = sky_rout - sky_rin = 4.1
def measure_photometry(image_file, positions, r=9.7, sky_rin=17.9, sky_rout=22, sky_width=4.1):
    with fits.open(image_file) as hdul:
        data = hdul[0].data
    apertures = CircularAperture(positions, r=r)
    annuli = CircularAnnulus(positions, r_in=sky_rin, r_out=sky_rout)
    raw_flux = aperture_photometry(data, apertures)
    sky_flux = aperture_photometry(data, annuli)
    back = sky_flux['aperture_sum'] / annuli.area
    net_flux = raw_flux['aperture_sum'] - back * apertures.area
    if np.any(net_flux <= 0):
        print(f"Negative or zero net flux detectec in {image_file}: {net_flux}")
        return None, None
    return net_flux, raw_flux['aperture_sum']

#Defines a function which performs differnetial photometry using comparison stars passed to it using comp_radec 
def differential_photometry(image_list, target_radec, comp_radec, aperture=5):
    target_fluxes, times = [], []
    comp_fluxes = [[] for _ in comp_radec]

    for img in image_list:
        print(f"\nProcessing image: {img}")
        target_xy = find_centroid_radec(img, *target_radec)
        comp_xy = [find_centroid_radec(img, *radec) for radec in comp_radec]

        # Check if any centroids failed
        if target_xy is None or any(c is None for c in comp_xy):
            print("Skipping image due to missing centroid(s)")
            continue

        all_xy = [target_xy] + comp_xy

        # Print centroids to inspect values
        print("Target XY:", target_xy)
        print("Comparison XYs:", comp_xy)

        #Load in net_flux from measure_photometry, skip over all invalid net_flux
        net_flux, _ = measure_photometry(img, all_xy, r=aperture)
        if net_flux is None:
            print(f"skipping {img} due to invalid flux")
            continue

        #Account for outliers by sigma-clipping our target and comparison fluxes, and skip over outlier fluxes
        target_flux_clipped = sigma_clip([net_flux[0]], sigma=3, maxiters=3)
        if target_flux_clipped.mask[0]:
            print(f"skipping outlier target flux in {img}: {target_flux}")
            debug_centroid(img, *target_radec, f"centroid_target{img.name}.png")
            continue
        target_flux = target_flux_clipped[0]
        comp_mean = np.mean(sigma_clip(net_flux[1:], sigma=3, maxiters=3))
        target_fluxes.append(target_flux / comp_mean)
        time = Time(fits.getheader(img)['DATE-OBS']).mjd
        times.append(time)

        #Append our comparison fluxes
        for i, f in enumerate(net_flux[1:]):
            comp_fluxes[i].append(f)

    return np.array(times), np.array(target_fluxes), np.array(comp_fluxes)

#A function which plots centroids for our target and comparison stars 
def debug_centroid(image_file, ra, dec, output="centroid_debug.png"):
    with fits.open(image_file) as hdul:
        data = hdul[0].data
        wcs = WCS(hdul[0].header)
    skycoord = SkyCoord(ra, dec, unit='deg')
    x, y = wcs.world_to_pixel(skycoord)
    norm = ImageNormalize(data, interval=ZScaleInterval(), stretch=LinearStretch())
    plt.imshow(data, origin='lower', norm=norm)
    plt.plot(x, y, 'rx', label='Initial (WCS)')
    cutout = data[int(y)-7:int(y)+8, int(x)-7:int(x)+8]
    mask, clean_data = detect_cosmics(cutout, sigclip=4.5, fsmode='median')
    dx, dy = centroid_com(clean_data) if not np.all(np.isnan(clean_data)) else (0,0)
    plt.plot(x - 7 + dx, y - 7 + dy, 'g+', label='Refined')
    plt.legend()
    plt.savefig(output, dpi=300)
    plt.close()
    print(f"Debug plot saved to {output}")

#A function which plots a light curve from times and diff_flux which were generated in differential_photometry
def plot_light_curves(times, diff_flux, output="lightcurve.png"):
    times = np.array(times)

    plt.figure(figsize=(8,5))
    plt.plot(times, diff_flux, marker='o')
    plt.xlabel("Time Since Start of Observation")
    plt.ylabel("Relative Flux Target / Comparison")
    plt.title("Differential Light Curve")
    plt.grid(True)
    plt.savefig(output, dpi=300)
    print(f"Light curve saved to {output}")

def plot_phase_curve(times, diff_flux, period, output="phase_curve.png"):
    times = np.array(times)
    diff_flux = np.array(diff_flux)

    #filter more invalid fluxes
    valid = (diff_flux > 0) & ~sigma_clip(diff_flux, sigma=3, maxiters=3).mask
    times = times[valid]
    diff_flux = diff_flux[valid]
    if len(times) == 0:
        print("No valid data points after filtering")

    # Convert to magnitudes
    mags = -2.5 * np.log10(diff_flux)

    #make sure times are in mjd
    times = Time(times, format='mjd')

    # Fixed T0 from Yang et al. (already in MJD)
    T0 = 54957.191639  

    # Phase calculation
    phase = ((times.mjd - T0) / period) % 1

    # Sort by phase
    sorted_idx = np.argsort(phase)
    phase = phase[sorted_idx]
    mags = mags[sorted_idx]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(phase, mags, 'o', markersize=4, label='Phase 0–1')
    plt.plot(phase + 1, mags, 'o', markersize=4, alpha=0.5, label='Phase 1–2')
    plt.xlabel("Phase")
    plt.ylabel("Magnitude")
    plt.title("Phase-folded Light Curve")
    plt.xlim(0, 2)
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.legend()
    plt.savefig(output, dpi=300)
    print(f"Phase curve saved to {output}")








    
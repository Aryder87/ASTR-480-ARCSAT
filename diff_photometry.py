import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.visualization import ZScaleInterval
from astropy.table import Table, vstack
from astropy.stats import sigma_clipped_stats
from photutils.aperture import CircularAperture, aperture_photometry, CircularAnnulus
from photutils.centroids import centroid_com
from astropy.coordinates import Angle
import glob

#Defines a function to calculate centroids given ra and dec using wcs and SkyCoord
def find_centroid_radec(image_file, ra, dec):
    with fits.open(image_file) as hdul:
        data = hdul[0].data
        wcs = WCS(hdul[0].header)
        skycoord = SkyCoord(ra, dec, unit='deg')
        x, y = wcs.world_to_pixel(skycoord)
        cutout = data[int(y)-10:int(y)+10, int(x)-10:int(x)+10] #accounts for an area around each coordinate
        dx, dy = centroid_com(cutout)
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

        net_flux, _ = measure_photometry(img, all_xy, r=aperture)
        target_flux = net_flux[0]
        comp_mean = np.mean(net_flux[1:])
        target_fluxes.append(target_flux / comp_mean)
        time = Time(fits.getheader(img)['DATE-OBS']).mjd
        times.append(time)

        for i, f in enumerate(net_flux[1:]):
            comp_fluxes[i].append(f)

    return np.array(times), np.array(target_fluxes), np.array(comp_fluxes)

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
    from astropy.time import Time
    import numpy as np
    import matplotlib.pyplot as plt

    times = np.array(times)
    times = Time(times, format='mjd')
    diff_flux = np.array(diff_flux)

    # Convert to magnitudes
    mags = -2.5 * np.log10(diff_flux)

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








    
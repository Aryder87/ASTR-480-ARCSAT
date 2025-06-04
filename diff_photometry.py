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


def find_centroid_radec(image_file, ra, dec):
    with fits.open(image_file) as hdul:
        data = hdul[0].data
        wcs = WCS(hdul[0].header)
        skycoord = SkyCoord(ra, dec, unit='deg')
        x, y = wcs.world_to_pixel(skycoord)
        cutout = data[int(y)-10:int(y)+10, int(x)-10:int(x)+10]
        dx, dy = centroid_com(cutout)
        return(x - 10 + dx, y - 10 + dy)
    
#sky_rin = 5.18 sky_rout = 10.37, sky_width = sky_rout - sky_rin
def measure_photometry(image_file, positions, r=5.41, sky_rin=5.18, sky_rout=10.37, sky_width=5.19):
    with fits.open(image_file) as hdul:
        data = hdul[0].data
    apertures = CircularAperture(positions, r=r)
    annuli = CircularAnnulus(positions, r_in=sky_rin, r_out=sky_rout)
    raw_flux = aperture_photometry(data, apertures)
    sky_flux = aperture_photometry(data, annuli)
    back = sky_flux['aperture_sum'] / annuli.area
    net_flux = raw_flux['aperture_sum'] - back * apertures.area
    return net_flux, raw_flux['aperture_sum']

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
    
def plot_light_curves(times, diff_flux, output="lightcurve.png"):
    times = np.array(times)
    time_hours = (times - times.min()) * 24 #converting to hours

    plt.figure(figsize=(8,5))
    plt.plot(times, diff_flux, marker='o')
    plt.xlabel("Time Since Start of Observation")
    plt.ylabel("Relative Flux Target / Comparison")
    plt.title("Differential Light Curve")
    plt.grid(True)
    plt.savefig(output, dpi=300)
    print(f"Light curve saved to {output}")






    
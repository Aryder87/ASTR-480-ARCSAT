import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.visualization import ZScaleInterval, ImageNormalize, LinearStretch
from astropy.stats import sigma_clip
from photutils.aperture import CircularAperture, aperture_photometry, CircularAnnulus
from photutils.centroids import centroid_com
import polars
import pathlib
import seaborn
import glob
import gc

#Defines a function to calculate centroids given ra and dec using wcs and SkyCoord
def find_centroid_pixel(image_file, x, y):
    with fits.open(image_file) as hdul:
        data = hdul[0].data
    print(f"Image {image_file}: Initial x={x:.2f}, y={y:.2f}, Shape={data.shape}")
    if not (0 <= int(x)-7 < data.shape[1] and 0 <= int(y)-7 < data.shape[0]): #ensures our centroid is in bound
        print(f"Initial centroid out of bounds for {image_file}: x={x}, y={y}")
        return None
    
    cutout = data[int(y)-7:int(y)+8, int(x)-7:int(x)+8] #accounts for an area around each coordinate (15x15 cutout)
    if np.any(np.isnan(cutout)) or np.all(cutout ==0):
        print(F"Invalid cutout (Nan or zero) for {image_file}: NaNs={np.sum(np.isnan(cutout))}, Zeros={np.all(cutout == 0)}")
        return None
    dx, dy = centroid_com(cutout)

    #cleanup
    del data, cutout
    gc.collect()
    
    return (x - 7 + dx, y - 7 + dy)

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
    
    #cleanup
    del data, sky_flux, back
    gc.collect()

    return net_flux, raw_flux['aperture_sum']

#Defines a function which performs differnetial photometry using comparison stars passed to it using comp_radec 
def differential_photometry(image_list, target_pix, comp_pix, aperture=5, save_npy=False):
    target_fluxes, times, target_flux_nocomp = [], [], []
    comp_fluxes = [[] for _ in comp_pix]

    for img in image_list:
        print(f"\nProcessing image: {img}")
        target_xy = find_centroid_pixel(img, *target_pix)
        comp_xy = [find_centroid_pixel(img, *pix) for pix in comp_pix]

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
            print(f"skipping outlier target flux in {img}")
            debug_centroid(img, *target_pix, f"centroid_target{img.name}.png")
            continue
        target_flux = target_flux_clipped[0]
        comp_mean = np.mean(sigma_clip(net_flux[1:], sigma=3, maxiters=3))
        target_fluxes.append(target_flux / comp_mean)
        target_flux_nocomp.append(target_flux)
        time = Time(fits.getheader(img)['DATE-OBS']).mjd
        times.append(time)

        #Append our comparison fluxes
        for i, f in enumerate(net_flux[1:]):
            comp_fluxes[i].append(f)

        #cleanup
        del net_flux, comp_mean, target_flux_clipped, target_flux
        gc.collect()

        if save_npy:
            np.save("target_flux_nocomp.npy", target_flux_nocomp)

    return np.array(times), np.array(target_fluxes), np.array(comp_fluxes), np.array(target_flux_nocomp)

#A function which plots centroids for our target and comparison stars 
def debug_centroid(image_file, x, y, output="centroid_debug.png"):
    with fits.open(image_file) as hdul:
        data = hdul[0].data

    norm = ImageNormalize(data, interval=ZScaleInterval(), stretch=LinearStretch())
    plt.imshow(data, origin='lower', norm=norm)
    plt.plot(x, y, 'rx', label='Initial (WCS)')
    cutout = data[int(y)-7:int(y)+8, int(x)-7:int(x)+8]
    dx, dy = centroid_com(cutout) if not np.all(np.isnan(cutout)) else (0,0)
    plt.plot(x - 7 + dx, y - 7 + dy, 'g+', label='Refined')
    plt.legend()
    plt.savefig(output, dpi=300)

    #cleanup
    plt.close('all')
    del data, cutout
    gc.collect()

    print(f"Debug plot saved to {output}")

#A function which plots a light curve from times and diff_flux which were generated in differential_photometry
def plot_light_curves(times, diff_flux, output="lightcurve.png"):
    times = np.array(times)
    diff_flux = np.array(diff_flux)

    df = polars.DataFrame({
        "times": times,
        "diff_flux": diff_flux
    })

    seaborn.set_theme(style="whitegrid")
    

    plt.figure(figsize=(8,5))
    seaborn.scatterplot(data=df, x='times', y='diff_flux')
    plt.xlabel("Time of Obsevation (MJD)")
    plt.ylabel("Relative Flux Target / Comparison")
    plt.title("Differential Light Curve")
    plt.tight_layout()
    plt.savefig(output, dpi=300)

    plt.close('all')
    print(f"Light curve saved to {output}")

def plot_phase_curve(times, diff_flux, period, output="phase_curve.png"):
    times = np.array(times)
    diff_flux = np.array(diff_flux)

    #filter more invalid fluxes
    valid = ~np.isnan(diff_flux) & (diff_flux > -1e10)
    times = times[valid]
    diff_flux = diff_flux[valid]
    if len(times) == 0:
        print("No valid data points after filtering")

    df = polars.DataFrame({
        "times": times,
        "diff_flux": diff_flux
    })

    #make sure times are in mjd
    times = Time(times, format='mjd')

    # Fixed T0 from Yang et al. (already in MJD)
    T0 = 54957.191639  

    # Convert to magnitudes & calculate phase
    df = df.with_columns([
        (-2.5 * np.log10(df["diff_flux"])).alias("mags"),
        (((df["times"] - T0) / period + 0.5) % 1).alias("phase_1")
    ])

    #insert phase 2 into our data fram
    df = df.with_columns([
        (df["phase_1"]+1).alias("phase_2")
    ])

    seaborn.set_theme(style="whitegrid")
    
    # Plot
    plt.figure(figsize=(8, 5))
    seaborn.scatterplot(data=df, x='phase_1', y='mags', label='Phase 0–1')
    seaborn.scatterplot(data=df, x='phase_2', y='mags', label='Phase 1–2')
    plt.xlabel("Phase")
    plt.ylabel("Magnitude")
    plt.title("Phase-folded Light Curve")
    plt.xlim(0, 2)
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.legend()
    plt.savefig(output, dpi=300)

    plt.close('all')
    print(f"Phase curve saved to {output}")

    


if __name__ == "__main__":

    reduced_dir = pathlib.Path("data").resolve()
    
    #Differential Photometry values LPSEB35	240.184(deg)	+43:08(deg)
    target_pix = (505.8, 503.7)
    target_pix = (505.8 - 100, 503.7 - 100) #account for wcs croppinig


    #Comparison stars ra and dec
    comp_pix = [(483.4, 618.8),
                (668.186, 204.731),
                (495.8, 752)]
    comp_pix = [(483.4 - 100, 618.8- 100),
                (668.186 - 100, 204.731 - 100),
                (495.8 - 100, 752 - 100)] #account for wcs cropping

    #define image_list and call on our reduced images
    image_list = sorted(pathlib.Path(reduced_dir).glob('reduced_science*_reprojected.fits'))

    #ensure images are being reprojected
    if not image_list:
        print(f"No reprojected images found in {reduced_dir} with pattern 'reduced_science*'")

    #call on time observed
    times, diff_flux, comp_fluxes, raw_flux = differential_photometry(image_list, target_pix, comp_pix, save_npy=True)

    #plot mag vs phase
    plot_phase_curve(times=times, diff_flux=diff_flux, period=0.25, output="phase_curve.png")

#Performs diagnostics on images to determine effects of air mass, 
#to explain a rise of flux > 1 in plots created by reduction/photometry and analysis

from astropy.io import fits 
from astropy.time import Time
from photutils.aperture import aperture_photometry, CircularAperture
import matplotlib.pyplot as plt
import seaborn as sb
import polars
import numpy as np
from astropy.stats import sigma_clipped_stats
import os 

def analyze_image_metadata(image_list, target_pix):
    airmass_list = []
    mjd_list = []
    fwhm_list = []
    background_list = []

    for filename in image_list:
        with fits.open(filename) as hdul:
            header = hdul[0].header
            data = hdul[0].data

            airmass = header.get("AIRMASS", np.nan)
            try:
                mjd = Time(header["DATE-OBS"], format = "isot", scale = "utc").mjd
            except Exception as e:
                mjd = np.nan

            airmass_list.append(airmass)
            mjd_list.append(mjd)

            try:
                _, median, _ = sigma_clipped_stats(data)
            except Exception:
                median = np.nan
            background_list.append(median)

            aperture = CircularAperture([target_pix], r=5)
            phot_table = aperture_photometry(data, aperture)
            flux = phot_table['aperture_sum'][0]

            #quick and dirty fwhm
            fwhm = 2.3548 * np.sqrt(flux) / flux if flux > 0 else np.nan
            fwhm_list.append(fwhm)
    
    return np.array(mjd_list), np.array(airmass_list), np.array(fwhm_list), np.array(background_list)

def plot_diagnostics(mjd, airmass, fwhm, background):

    mask= ~np.isnan(mjd) & ~np.isnan(airmass) & ~np.isnan(fwhm) & ~np.isnan(background)

    df = polars.DataFrame({
        "MJD": mjd[mask],
        "Airmass": airmass[mask],
        "FWHM": fwhm[mask],
        "SkyBack": background[mask]
    }).sort("MJD")

    sb.set_theme(style="whitegrid")

    fig, axs = plt.subplots(3, 1, figsize=(10,8), sharex=True)

    sb.scatterplot(data=df, x="MJD", y="Airmass", ax=axs[0], s=40, alpha=0.6)
    sb.regplot(data=df, x="MJD", y="Airmass", ax=axs[0], scatter=False, lowess=True, color="darkblue")
    axs[0].set_ylabel("Airmass")
    axs[0].set_title("Airmass over Time")

    sb.scatterplot(data=df, x="MJD", y="FWHM", ax=axs[1], s=40, alpha=0.6)
    sb.regplot(data=df, x="MJD", y="FWHM", ax=axs[1], scatter=False, lowess=True, color="darkgreen")
    axs[1].set_ylabel("FWHM Estimate")
    axs[1].set_title("Seeing (FWHM proxy)")

    sb.scatterplot(data=df, x="MJD", y="SkyBack", ax=axs[2], s=40, alpha=0.6)
    sb.regplot(data=df, x="MJD", y="SkyBack", ax=axs[2], scatter=False, lowess=True, color="darkred")
    axs[2].set_ylabel("Background level")
    axs[2].set_xlabel("MJD")
    axs[2].set_title("Sky Background over Time")

    ecl_start = 60825.18
    ecl_end   = 60825.23

    for ax in axs:
        ax.axvspan(ecl_start, ecl_end, color='orange', alpha=0.2, label='Eclipse')

    plt.tight_layout()
    plt.savefig("diagnostics.png")
    plt.close()

if __name__ == "__main__":
    from pathlib import Path

    image_dir = Path("/Users/ryder47/ASTR-480-ARCSAT/data/")
    image_list = sorted(image_dir.glob("reduced_science*.fits"))

    target_pix = (505.8 - 100, 503.7 - 100)

    mjd, airmass, fwhm, background = analyze_image_metadata(image_list, target_pix)
    plot_diagnostics(mjd, airmass, fwhm, background)


    
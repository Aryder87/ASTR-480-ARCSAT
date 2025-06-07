#Write code to match the wcs of each image of LPSEB to align them 
#Then images are aligned, import them to reduction, and change photometry
#To get the stars based off of ra and dec, also see if the previous can be done without aligning the images

from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
import numpy as np 
from matplotlib import pyplot as plt
import glob
import pathlib
import gc
import subprocess
import os


"""If you are on Mac: install astrometry-net via brew

`brew install astrometry-net`

"""
def solve_wcs_astrometry(image_path):
    try:
        result = subprocess.run([
            "solve-field",
            "--overwrite",
            "--no-plots",
            "--crpix-center",
            "--scale-units", "arcsecperpix",
            "--scale-low", "0.3",
            "--scale-high", "2.5",
            "--downsample", "2",
            image_path
        ], check=True)

        solved_img = str(image_path).replace(".fits", "-wcs.fits")
        if os.path.exists(solved_img):
            return solved_img
        else:
            return None
    except subprocess.CalledProcessError as e:
        print(f"[x] solve-field failed for {image_path}: {e}")
        return None

def center_image(reduced_dir):

    #Ensure that data_dir is a pathlib object
    reduced_dir = pathlib.Path(reduced_dir).resolve()

    #Load in the list of images
    image_list = sorted(pathlib.Path(reduced_dir).glob('reduced_science*.fits'))

    #load in our image used for centering 
    reference_file = reduced_dir / 'reduced_science1.fits'
    try:
        with fits.open(reference_file) as goodpoint:
            ref_header = goodpoint[0].header.copy()
    except FileNotFoundError:
        print(f"Reference file {reference_file} not found.") #make sure our image is here and found! 
        return
    
    for filename in image_list:
        try:
            with fits.open(filename) as hdul:
                input_hdu = hdul[0]
                original_date_obs = input_hdu.header.get('DATE-OBS')
                reprojected, footprint = reproject_interp(input_hdu, ref_header)
        except Exception as e:
            print(f"[!] Reprojection failed for {filename}: {e}\n  Attemping astrometry.net solution...")
            solved_img = solve_wcs_astrometry(filename)
            if not solved_img:
                print(f"[x] Astrometry solution failed for {filename}")
                continue

            with fits.open(solved_img) as hdul:
                input_hdu = hdul[0]
                original_date_obs = input_hdu.header.get('DATE-OBS')
                try:
                    reprojected, footprint = reproject_interp(input_hdu, ref_header)
                except Exception as e:
                    print(f"[x] Reprojection still failed after WCS solve: {e}")
                    continue
        new_header = ref_header.copy()
        if original_date_obs:
            new_header["DATE-OBS"] = original_date_obs

        hdu = fits.PrimaryHDU(reprojected, header=new_header)
        output_filename = f"{reduced_dir}/{filename.stem}_reprojected.fits"
        hdu.writeto(output_filename, overwrite=True)

        print(f"Saved reprojected image to {output_filename}")

        del reprojected, footprint, hdu, input_hdu  # Free memory
        gc.collect()

    goodpoint.close()



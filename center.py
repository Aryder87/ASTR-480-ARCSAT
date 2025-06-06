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


    # Assume you have:
    # image_list = list of file paths (e.g., ["img1.fits", "img2.fits", ...])
    # wcs_good = WCS object for the target projection
    # goodpoint = FITS file opened with fits.open(), whose header defines the target WCS

    for filename in image_list:
        with fits.open(filename) as hdul:
            # Try hdul[0] — change to [1] if your data is in an extension
            input_hdu = hdul[0]

            #need to preserve the original observation time
            original_date_obs = input_hdu.header.get('DATE-OBS')   
            try:
                reprojected, footprint = reproject_interp(input_hdu, ref_header)
            except ValueError as e:
                print(f"Skipping {filename}: {e}")
                continue

            new_header = ref_header.copy()
            if original_date_obs:
                new_header["DATE-OBS"] = original_date_obs
            
            hdu = fits.PrimaryHDU(reprojected, header=new_header)
            output_filename = f"{filename.stem}_reprojected.fits"
            hdu.writeto(output_filename, overwrite=True)

            print(f"Saved reprojected image to {output_filename}")

            del reprojected, footprint, hdu, input_hdu  # Free memory
            gc.collect()

    goodpoint.close()



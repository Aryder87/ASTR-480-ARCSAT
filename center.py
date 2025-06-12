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

Mak sure to go into the astrometry/data to download the necessary index files to 
solve for the images that could not be solved.

"""
def solve_wcs_astrometry(image_path, index_dir="astrometry/index"):
    """
    Solve the WCS using astrometry.net solve-field, with index_dir = 
    location the downloaded index files
    """

    #Since I know the only index files are downloaded in this directory, I set it here
    index_dir = pathlib.Path(index_dir).resolve()
    os.environ["ASTROMETRY_INDEX_DIR"] = str(index_dir)

    #Skip over images whose wcs has already been solved
    cached_solution = image_path.parent / (image_path.stem + ".new")
    if cached_solution.exists():
        print(f"[i] Skipping over {image_path.name}, there is already a solved wcs")
        return cached_solution

    cmd = [
            "solve-field",
            "--overwrite",
            "--no-plots",
            "--crpix-center",
            "--scale-units", "arcsecperpix",
            "--scale-low", "0.3",
            "--scale-high", "2.0",
            "--downsample", "2",
            "--depth", "20",
            "--index-dir", str(pathlib.Path(index_dir)),
            str(image_path)
    ]

    
    try:
        result = subprocess.run(cmd, check=True, cwd=image_path.parent)
        if cached_solution.exists():
            return cached_solution
        else:
            print(f"WCS has not been solved for {image_path.name}, will solve now")
            return None
        
    except subprocess.CalledProcessError as e:
        print(f"[x] solve-field failed for {image_path}: {e}")
        return None
    
#Created to delete unnecessary files creating using astrometry and clear space 
def cleanup_astrometry_files(base_path):
    """
    Deletes auxiliary files created by solve-field for a given base image.
    """
    suffixes = [
        ".axy", ".corr", ".match", ".rdls", ".solved",
        ".wcs", ".fits", ".png", "-indx.xyls"
    ]
    for suffix in suffixes:
        pattern = base_path.parent / f"{base_path.stem}{suffix}"
        for f in base_path.parent.glob(pattern.name):
            try:
                f.unlink()
            except Exception as e:
                print(f"[!] Could not delete {f}: {e}")

def center_image(reduced_dir, index_dir="astrometry/data"):
    """
    Don't forget to call your index dir! 
    """

    #Ensure that data_dir is a pathlib object
    reduced_dir = pathlib.Path(reduced_dir).resolve()

    #Load in the list of images
    image_list = sorted(pathlib.Path(reduced_dir).glob('reduced_science*.fits'))

    #Solving WCS for ALL images using astrometry for robustness
    print("[i] solving WCS for all images...")
    solved_images = []
    for img in image_list:
        solved = solve_wcs_astrometry(img, index_dir=index_dir)
        if solved:
            solved_images.append(solved)
        else:
            print(f"[x] could not solve wcs for {img}")
    
    #Choose our reference image, and ensure images were solved
    if not solved_images:
        print("[x] No images were solved; cannot continue anymore boss")
        return
    
    ref_file = solved_images[0]
    print(f"[i] Using {ref_file} as WCS ref.")

    with fits.open(ref_file) as ref_hdul:
        ref_header = ref_hdul[0].header.copy()

    #Now reproject or skip images that were already reprojected
    for solved_img in solved_images:
        out_path = reduced_dir / f"{solved_img.stem}_reprojected.fits"

        if out_path.exists():
            print(f"[i] Skipping reprojection for {solved_img.name}; already exists.")
            continue

        with fits.open(solved_img) as hdul:
            input_hdu = hdul[0]
            original_date_obs = input_hdu.header.get('DATE-OBS')
            try:
                reprojected, footprint = reproject_interp(input_hdu, ref_header)
            except Exception as e:
                print(f"[x] Reprojection failed for {solved_img.name}: {e}")
                continue
    
        new_header = ref_header.copy()
        if original_date_obs:
            new_header["DATE-OBS"] = original_date_obs
        
        fits.PrimaryHDU(reprojected, header=new_header).writeto(out_path, overwrite=True)
        print(f"[i] Saved reprojected image to {out_path.name}")

        del reprojected, footprint, input_hdu
        gc.collect()

        cleanup_astrometry_files(solved_img)

    print("All images aligned! Good work boss")



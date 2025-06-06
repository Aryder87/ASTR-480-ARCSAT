from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import numpy as np

def radec_to_pixel(image_file, radec_list):
    with fits.open(image_file) as hdul:
        wcs = WCS(hdul[0].header)
    
    pixel_coords = []
    for i, (ra, dec) in enumerate(radec_list):
        skycoord = SkyCoord(ra, dec, unit='deg')
        x, y = wcs.world_to_pixel(skycoord)
        label = "Target" if i == 0 else f"Comp {i}"
        print(f"{label}: x={x:.1f}, y={y:.1f}")
        pixel_coords.append((x, y))

    return pixel_coords
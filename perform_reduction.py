import numpy as np
import polars
import seaborn
import glob
import gc

from reduction import run_reduction

# Main function

if __name__ == "__main__":
    data_dir = "data/"

    run_reduction(data_dir, skip=True, save_npy=True)


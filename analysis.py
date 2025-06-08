import numpy as np
import polars
import seaborn
import glob
import gc
import matplotlib.pyplot as plt

from astropy.timeseries import LombScargle
import astropy.units as u

def determine_lc_period(times, fluxes, plot=False):
    frequency, power = LombScargle(times, fluxes).autopower()

    best_freq = frequency[np.argmax(power)]

    if not hasattr(best_freq, 'unit'):
        best_freq = best_freq / u.day

    period = 1/best_freq

    # The actual period is twice the period of the periodiogram
    lc_period = period.to(u.h) * 2

    if plot:
        plt.plot(frequency, power)
        plt.axvline(x=best_freq.value, label=period, c='orange')
        plt.xlabel("Freq (1/day)")
        plt.title(f"System period: {lc_period}")
        plt.xlim(0, 50)
        plt.legend()
        plt.show()
        plt.clf() 

    return lc_period

if __name__ == "__main__":

    # Load the datas
    times = np.load("times.npy") * u.day
    fluxes = np.load("diff_flux.npy")    

    lc_period = determine_lc_period(times, fluxes, plot=True)
    print(lc_period)


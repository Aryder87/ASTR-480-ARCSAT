import numpy as np
import polars
import seaborn
import glob
import gc
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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
        plt.savefig("Lomb Scargle frequency plot")
        plt.clf() 
        plt.close()

    return lc_period

#Calculate phase folding time using times found in diff_phot
def fold_time(times, period = 0.25, T0 = 54957.191639):
    #T0 from Yang et al. 2009
    #calculating phase from known period and times found in diff_phot
    phase = ((times - T0) / period) % 1 
    phase[phase > 0.5] -= 1   #recenter phase
    return phase

#Define a function which takes scipy fitted trapezoid and models and 
#calculates depth, duration, and ingress 
def trapezoid_model(phase, mid, depth, duration, ingress):
    flux = np.ones_like(phase)
    ingress_start = mid - duration / 2
    ingress_end = mid - duration / 2 + ingress
    egress_start = mid + duration / 2 - ingress
    egress_end = mid + duration / 2

    for i, ph in enumerate(phase):
        if ingress_start <= ph < ingress_end:
            flux[i] = 1 - depth * (ph - ingress_start) / ingress
        elif ingress_end <= ph < egress_start:
            flux[i] = 1 - depth
        elif egress_start <= ph < egress_end:
            flux[i] = 1 - depth * (1 - (ph - egress_start) / ingress)
    return flux

def fit_trapezoids(times, fluxes, period = 0.25, T0 = 54957.191639, plot=True):
    phase = fold_time(times, period, T0)
    sort = np.argsort(phase)
    phase, flux = phase[sort], fluxes[sort]

    #guess numbers for our trapezoid_model, i.e. center, mid, depth, ingress_egress
    guess_primary = [0.0, 0.5, 0.1, 0.01] #we know depth is ~ 0.75 mag (Yang et al. 2009) = 0.5
    guess_secondary = [0.3, 0.5, 0.04, 0.01]

    mask_primary = np.abs(phase) < 0.15
    mask_secondary = np.abs(phase - 0.5) < 0.3

    popt1, pcov1 = curve_fit(
        trapezoid_model, phase[mask_primary], flux[mask_primary],
        p0 = guess_primary
    )
    popt2, pcov2 = curve_fit(
        trapezoid_model, phase[mask_secondary], flux[mask_secondary],
        p0 = guess_secondary
    )

    if plot:
        phase_fit = np.linspace(-0.6, 0.6, 1000)
        plt.figure(figsize=(10,5))
        plt.scatter(phase, flux, s=10, label='Data', alpha=0.6)
        plt.plot(phase_fit, trapezoid_model(phase_fit, *popt1), label='Primary Fit', color='red')
        plt.plot(phase_fit, trapezoid_model(phase_fit, *popt2), label='Secondary Fit', color='blue')
        plt.xlabel('Phase')
        plt.ylabel('Relative Flux')
        plt.title('Trapezoidal Eclipse Fits')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig("eclipse_trapezoid_fit.png")
        plt.close()
    
    return{
        "primary": {
            "mid_phase": popt1[0],
            "mid_time_mjd": T0 + period * popt1[0],
            "depth": popt1[1],
            "duration_days": period * popt1[2],
        },
        "secondary": {
            "mid_phase": popt2[0],
            "mid_time_mjd": T0 + period * popt2[0],
            "depth": popt2[1],
            "duration_days": period * popt2[2],
        }
    }

def calc_ingress_egress(eclipse_params, period_days):
    for eclipse_name, params in eclipse_params.items():
        mid_phase = params['mid_phase']
        mid_time_mjd = params['mid_time_mjd']
        duration_days = params['duration_days']

        duration_phase = duration_days / period_days

        ingress_phase = (mid_phase - duration_phase / 2) % 1
        egress_phase = (mid_phase + duration_phase / 2) % 1

        ingress_mjd = mid_time_mjd - duration_days / 2
        egress_mjd = mid_time_mjd + duration_days / 2

        print(f"{eclipse_name.capitalize()} Eclipse:")
        print(f" Ingress phase: {ingress_phase:.5f}")
        print(f" Egress phase: {egress_phase:.5f}")
        print(f" Ingress MJD: {ingress_mjd:.5f}")
        print(f" Egress MJD: {egress_mjd:.5f}")
        print()

if __name__ == "__main__":

    # Load the datas
    times = np.load("times.npy") 
    fluxes = np.load("diff_flux.npy")    

    lc_period = determine_lc_period(times, fluxes, plot=True)
    print(lc_period)

    period_days = lc_period.to(u.day).value

    fitted_traps = fit_trapezoids(times, fluxes, period=period_days, T0 = 54957.191639, plot=True)
    print("Fitted Trapezoid parameters")
    print(fitted_traps)

    print("\nIngress and Egress times")
    calc_ingress_egress(fitted_traps, period_days)

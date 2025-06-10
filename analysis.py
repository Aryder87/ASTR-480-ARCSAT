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
    guess_primary = [0.3, 0.5, 0.2, 0.01] #we know depth is ~ 0.75 mag (Yang et al. 2009) = 0.5
    mask_primary = np.abs(phase - guess_primary[0]) < 0.5

    popt1, pcov1 = curve_fit(
        trapezoid_model, phase[mask_primary], flux[mask_primary],
        p0 = guess_primary
    )

    if plot:
        phase_fit = np.linspace(-0.5, 0.8, 1000)
        plt.figure(figsize=(10,5))
        plt.scatter(phase, flux, s=10, label='Data', alpha=0.6)
        plt.plot(phase_fit, trapezoid_model(phase_fit, *popt1), label='Primary Fit', color='red')
        plt.xlabel('Phase')
        plt.ylabel('Relative Flux')
        plt.xlim(-0.1, 0.8)
        plt.title('Trapezoidal Eclipse Fits')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.close()
    
    ingress_min = popt1[3] * period * 24 * 60 
    egress_min = ingress_min

    return{
        "primary": {
            "mid_phase": popt1[0],
            "mid_time_mjd": T0 + period * popt1[0],
            "depth": popt1[1],
            "duration_days": period * popt1[2],
            "ingress_minutes": ingress_min,
            "egress_minutes": egress_min,
        }
    }

def calc_ingress_egress(eclipse_params, period_days):
    for eclipse_name, params in eclipse_params.items():
        mid_phase = params['mid_phase']
        mid_time_mjd = params['mid_time_mjd']
        duration_days = params['duration_days']

        duration_phase = duration_days / period_days

        ingress_phase = (mid_phase - duration_phase / 2) 
        egress_phase = (mid_phase + duration_phase / 2) 

        ingress_mjd = mid_time_mjd - duration_days / 2
        egress_mjd = mid_time_mjd + duration_days / 2

        print(f"{eclipse_name.capitalize()} Eclipse:")
        print(f" Ingress phase: {ingress_phase:.5f}")
        print(f" Egress phase: {egress_phase:.5f}")
        print(f" Ingress MJD: {ingress_mjd:.5f}")
        print(f" Egress MJD: {egress_mjd:.5f}")
        print()

def trapezoid_in_time_model(times, mid_time, depth, duration, ingress):
    flux = np.ones_like(times)
    ingress_start = mid_time - duration / 2
    ingress_end = ingress_start + ingress
    egress_start = mid_time + duration / 2 - ingress
    egress_end = mid_time + duration / 2

    ramp_up = (ingress_start <= times) & (times < ingress_end)
    flat = (ingress_end <= times) & (times < egress_start)
    ramp_down = (egress_start <= times) & (times < egress_end)

    flux[ramp_up] = 1 - depth * (times[ramp_up] - ingress_start) / ingress
    flux[flat] = 1 - depth
    flux[ramp_down] = 1 - depth * (1 - (times[ramp_down] - egress_start) / ingress)

    return flux


def plot_phase_and_time_fits(times, fluxes, period_days, 
                             T0, phase_fit_params, time_fit_params):
    # Phase-domain plot
    phase = fold_time(times, period_days, T0)
    sorted_phase_idx = np.argsort(phase)
    phase_sorted = phase[sorted_phase_idx]
    flux_sorted = fluxes[sorted_phase_idx]

    phase_model = np.linspace(-0.5, 0.8, 1000)
    phase_model_flux = trapezoid_model(
        phase_model,
        phase_fit_params['mid_phase'],
        phase_fit_params['depth'],
        phase_fit_params['duration_days'] / period_days,
        phase_fit_params['ingress_minutes'] / (24 * 60 * period_days)
    )

    tmin, tmax = times.min(), times.max()
    tgrid = np.linspace(tmin, tmax, 500)

    ingress_days = time_fit_params["ingress_minutes"] / (24*60)

    model_grid = trapezoid_in_time_model(
        tgrid,
        time_fit_params["mid_time_mjd"],
        time_fit_params["depth"],
        time_fit_params["duration_days"],
        ingress_days
    )

    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Phase-domain
    axs[0].scatter(phase_sorted, flux_sorted, s=5, alpha=0.5, label='Observed')
    axs[0].plot(phase_model, phase_model_flux, color='red', lw=2, label='Phase Trapezoid')
    axs[0].set_xlabel("Phase")
    axs[0].set_ylabel("Flux")
    axs[0].set_xlim(-0.1, 0.8)
    axs[0].set_title("Phase-Domain Trapezoid Fit")
    axs[0].legend()
    axs[0].grid(True)

    # Time-domain
    axs[1].scatter(times, fluxes, s=5, alpha=0.5, label='Observed')
    axs[1].plot(tgrid, model_grid, color='green', lw=2, label='Time Trapezoid')
    axs[1].set_xlabel("Time (MJD)")
    axs[1].set_ylabel("Flux")
    axs[1].set_title("Time-Domain Trapezoid Fit")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig("trapezoid_phase_vs_time_fit.png")
    plt.close()


def fit_time_domain_trapezoid(times, fluxes, period, T0, trap_phase_fit):
    # Convert trapezoid phase-fit parameters to time-based initial guesses
    mid_phase = trap_phase_fit["mid_phase"]
    depth = trap_phase_fit["depth"]
    duration_days = trap_phase_fit["duration_days"]
    ingress_minutes = trap_phase_fit["ingress_minutes"]
    ingress_days = ingress_minutes / (24 * 60)

    # Estimate mid-eclipse time based on reference T0
    mid_time_guess = T0 + mid_phase * period

    # Initial parameter guess: [mid_time, depth, duration_days, ingress_days]
    p0 = [mid_time_guess, depth, duration_days, ingress_days]

    print("\n[DEBUG] Initial trapezoid parameter guess:")
    print(f" Mid time: {mid_time_guess:.5f}")
    print(f" Depth: {depth:.3f}")
    print(f" Duration: {duration_days:.5f}")
    print(f" Ingress: {ingress_days:.5f}")

    try:
        popt, pcov = curve_fit(trapezoid_in_time_model, times, fluxes, p0=p0)
        mid_time, depth_fit, duration_fit, ingress_fit = popt
        ingress_minutes_fit = ingress_fit * 24 * 60

        print("\nTime-domain Trapezoid Fit:")
        print(f"  Mid-eclipse Time (MJD): {mid_time:.6f}")
        print(f"  Depth: {depth_fit:.3f}")
        print(f"  Duration: {duration_fit:.4f} days")
        print(f"  Ingress/Egress: {ingress_minutes_fit:.2f} minutes")

        return {
            "mid_time_mjd": mid_time,
            "depth": depth_fit,
            "duration_days": duration_fit,
            "ingress_minutes": ingress_minutes_fit,
        }
    except RuntimeError:
        print(" Fit did not converge.")
        return None

if __name__ == "__main__":

    times = np.load("times.npy") 
    fluxes = np.load("diff_flux.npy") 
    fluxes = fluxes / np.median(fluxes)   

    lc_period = determine_lc_period(times, fluxes, plot=True)
    print(lc_period)
    period_days = lc_period.to(u.day).value

    # First pass: crude T0
    T0_initial = 54957.191639  # or use times.min() if you prefer

    fitted_traps = fit_trapezoids(times, fluxes, period=period_days, T0=T0_initial, plot=True)

    # Now estimate better T0 from min flux
    T0_guess = times[np.argmin(fluxes)] - fitted_traps["primary"]["mid_phase"] * period_days
    print(f"[INFO] Estimated modern T0 from min flux: {T0_guess:.5f}")

    # Refit with improved T0
    fitted_traps = fit_trapezoids(times, fluxes, period=period_days, T0=T0_guess, plot=True)

    trap_time_fit = fit_time_domain_trapezoid(
        times, fluxes, period_days, T0=T0_guess, trap_phase_fit=fitted_traps["primary"]
    )

    if trap_time_fit:
        plot_phase_and_time_fits(
            times, fluxes,
            period_days, T0=T0_guess,
            phase_fit_params=fitted_traps["primary"],
            time_fit_params=trap_time_fit
        )

    print("Fitted Trapezoid parameters")
    print(fitted_traps)

    print("Ingress and Egress durations (minutes):")
    print(f" Ingress duration: {fitted_traps['primary']['ingress_minutes']:.2f} minutes")
    print(f" Egress duration: {fitted_traps['primary']['egress_minutes']:.2f} minutes")

    print("\nIngress and Egress times")
    print("Flux min/max:", fluxes.min(), fluxes.max())

    calc_ingress_egress(fitted_traps, period_days)

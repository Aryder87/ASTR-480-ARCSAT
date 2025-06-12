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

def fit_trapezoids(times, fluxes, period = 0.25, T0 = 54957.191639, plot=False):
    phase = fold_time(times, period, T0)
    sort = np.argsort(phase)
    phase, flux = phase[sort], fluxes[sort]

    #guess numbers for our trapezoid_model, i.e. center, mid, depth, ingress_egress
    guess_primary = [0.3, 0.5, 0.2, 0.02] #we know depth is ~ 0.75 mag (Yang et al. 2009) = 0.5
    mask_primary = np.abs(phase - guess_primary[0]) < 0.5

    popt1, pcov1 = curve_fit(
        trapezoid_model, phase[mask_primary], flux[mask_primary],
        p0 = guess_primary
    )
    
    ingress_min = popt1[3] * period * 24 * 60 
    egress_min = ingress_min

    # Compute ingress and egress phases
    ingress_phase = popt1[0] - popt1[2]/2
    egress_phase = popt1[0] + popt1[2]/2

    return{
        "primary": {
            "mid_phase": popt1[0],
            "mid_time_mjd": T0 + period * popt1[0],
            "depth": popt1[1],
            "duration_days": period * popt1[2],
            "ingress_minutes": ingress_min,
            "egress_minutes": egress_min,
            "ingress_phase": ingress_phase,
            "egress_phase": egress_phase,
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

if __name__ == "__main__":

    times = np.load("times.npy") 
    fluxes = np.load("diff_flux.npy") 

    # Quick diagnostic: estimate eclipse depth directly from raw fluxes
    mag_depth_direct = -2.5 * np.log10(np.min(fluxes) / np.max(fluxes))
    print(f"[Direct from data] Estimated mag drop: {mag_depth_direct:.3f} mag")   

    lc_period = determine_lc_period(times, fluxes, plot=True)
    print(lc_period)
    period_days = lc_period.to(u.day).value

    # First pass: crude T0
    T0_initial = 54957.191639  # or use times.min() if you prefer

    #fit trapezoid before normalizing to get a better T0 guess/mid phase
    fitted_traps = fit_trapezoids(times, fluxes, period=period_days, T0=T0_initial, plot=True)

    # Now estimate better T0 from min flux
    T0_guess = times[np.argmin(fluxes)] - fitted_traps["primary"]["mid_phase"] * period_days
    print(f"[INFO] Estimated modern T0 from min flux: {T0_guess:.5f}")

    # Normalize using only out-of-eclipse baseline
    phase = fold_time(times, period_days, T0_guess)
    mid = fitted_traps["primary"]["mid_phase"]
    dur = fitted_traps["primary"]["duration_days"] / period_days

    # Mask out eclipse
    mask_out_of_eclipse = np.abs(phase - mid) > dur / 2 + 0.05

    # Normalize only to baseline
    baseline_median = np.median(fluxes[mask_out_of_eclipse])
    fluxes = fluxes / baseline_median

    # Refit with improved T0 and normalized fluxes
    fitted_traps = fit_trapezoids(times, fluxes, period=period_days, T0=T0_guess, plot=True)

    #Recompute phase for plotting, in eclipse times 
    phase = fold_time(times, period_days, T0_guess)
    sort = np.argsort(phase)
    phase_sorted = phase[sort]
    flux_sorted = fluxes[sort]

    # Model curve
    phase_model = np.linspace(-0.1, 0.8, 1000)
    model_flux = trapezoid_model(
        phase_model,
        fitted_traps["primary"]["mid_phase"],
        fitted_traps["primary"]["depth"],
        fitted_traps["primary"]["duration_days"] / period_days,
        fitted_traps["primary"]["ingress_minutes"] / (24 * 60 * period_days)
    ) 

    # Plot clean phase model with ingress/egress durations
    plt.figure(figsize=(10, 5))
    plt.scatter(phase_sorted, flux_sorted, s=10, alpha=0.6, label='Data')
    plt.plot(phase_model, model_flux, color='red', label='Trapezoid Model')

    # Compute trapezoid edges
    duration = fitted_traps["primary"]["duration_days"] / period_days
    ingress_duration = fitted_traps["primary"]["ingress_minutes"] / (24 * 60 * period_days)

    ingress_start = fitted_traps["primary"]["mid_phase"] - duration/2
    ingress_end   = ingress_start + ingress_duration
    egress_end    = fitted_traps["primary"]["mid_phase"] + duration/2
    egress_start  = egress_end - ingress_duration

    # Shade ingress and egress
    plt.axvspan(ingress_start, ingress_end, color='orange', alpha=0.2, label='Ingress Zone')
    plt.axvspan(egress_start, egress_end, color='purple', alpha=0.2, label='Egress Zone')

    # Add arrows or text for clarity (optional)
    plt.text(ingress_start, 1.01, f"Start", ha='center', fontsize=8, color='darkorange')
    plt.text(ingress_end, 1.01, f"End", ha='center', fontsize=8, color='darkorange')
    plt.text(egress_start, 1.01, f"Start", ha='center', fontsize=8, color='purple')
    plt.text(egress_end, 1.01, f"End", ha='center', fontsize=8, color='purple')

    # Main plot formatting
    plt.xlabel("Phase")
    plt.ylabel("Relative Flux")
    plt.xlim(-0.1, 0.7)
    plt.ylim(0.65, 1.15)
    plt.title("Trapezoidal Eclipse Fit (with Ingress/Egress Zones)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Trapezoidal_Eclipse_Fit_Normalized.png", dpi=300)
    plt.close()

    duration_days = fitted_traps["primary"]["duration_days"]
    duration_minutes = duration_days * 24 * 60 
    duration_hours = duration_days * 24

    print(f"\n Eclipse Duration:")
    print(f" In days:   {duration_days:.5f}")
    print(f" In hours:  {duration_hours:.2f}")
    print(f" In minutes:{duration_minutes:.2f}")

    print("Fitted Trapezoid parameters")
    print(fitted_traps)

    print("Ingress and Egress durations (minutes):")
    print(f" Ingress duration: {fitted_traps['primary']['ingress_minutes']:.2f} minutes")
    print(f" Egress duration: {fitted_traps['primary']['egress_minutes']:.2f} minutes")

    print("\nIngress and Egress times")
    print("Flux min/max:", fluxes.min(), fluxes.max())

    calc_ingress_egress(fitted_traps, period_days)

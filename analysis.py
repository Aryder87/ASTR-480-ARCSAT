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
    lc_period = period.to(u.h) * 2
    if plot:
        plt.plot(frequency, power)
        plt.axvline(x=best_freq.value, label=period, c='orange')
        plt.xlabel("Freq (1/day)")
        plt.title(f"System period: {lc_period}")
        plt.xlim(0, 50); plt.legend()
        plt.savefig("Lomb_Scargle_frequency_plot.png"); plt.close()
    return lc_period

def fold_time(times, period=0.25, T0=54957.191639):
    phase = ((times - T0) / period) % 1
    phase[phase > 0.5] -= 1  # recenter
    return phase

def trapezoid_model(phase, mid, depth, duration, ingress):
    flux = np.ones_like(phase)
    i0, i1 = mid - duration/2, mid - duration/2 + ingress
    e0, e1 = mid + duration/2 - ingress, mid + duration/2
    for i, ph in enumerate(phase):
        if   i0 <= ph < i1: flux[i] = 1 - depth*(ph - i0)/ingress
        elif i1 <= ph < e0: flux[i] = 1 - depth
        elif e0 <= ph < e1: flux[i] = 1 - depth*(1 - (ph - e0)/ingress)
    return flux

def fit_trapezoids(times, fluxes, period=0.25, T0=54957.191639):
    phase = fold_time(times, period, T0)
    idx = np.argsort(phase)
    phase, flux = phase[idx], fluxes[idx]
    guess = [0.3, 0.5, 0.2, 0.02]
    mask = np.abs(phase - guess[0]) < 0.5
    popt, _ = curve_fit(trapezoid_model, phase[mask], flux[mask], p0=guess)
    mid, depth, dur_ph, ing_ph = popt
    ingress_min = ing_ph * period * 24*60
    return {
        "primary": {
            "mid_phase":   mid,
            "mid_time_mjd": T0 + period*mid,
            "depth":       depth,
            "duration_days": period * dur_ph,
            "ingress_minutes": ingress_min,
            "egress_minutes":  ingress_min,
        }
    }

def calc_ingress_egress(params, period_days):
    p = params["primary"]
    mid = p["mid_phase"]; dur = p["duration_days"]
    dp = dur/period_days
    print("Ingress phase:", mid-dp/2)
    print("Egress phase: ", mid+dp/2)
    print("Ingress MJD:  ", p["mid_time_mjd"]-dur/2)
    print("Egress MJD:   ", p["mid_time_mjd"]+dur/2)

if __name__ == "__main__":
    times  = np.load("times.npy")
    fluxes = np.load("diff_flux.npy")

    # 1) determine period
    lc_period = determine_lc_period(times, fluxes)

    # 2) initial trapezoid fit
    fitted = fit_trapezoids(times, fluxes, period=lc_period.to(u.day).value)

    # 3) Normalize around out-of-eclipse baseline
    period_days = lc_period.to(u.day).value
    T0_guess   = times[np.argmin(fluxes)] - fitted["primary"]["mid_phase"]*period_days
    phase      = fold_time(times, period_days, T0_guess)
    mid        = fitted["primary"]["mid_phase"]
    dur_ph     = fitted["primary"]["duration_days"]/period_days
    mask_base  = np.abs(phase-mid) > dur_ph/2 + 0.05
    fluxes     = fluxes/np.median(fluxes[mask_base])

    # 4) refit on normalized data
    fitted = fit_trapezoids(times, fluxes, period=period_days)
    mid   = fitted["primary"]["mid_phase"]
    depth = fitted["primary"]["depth"]
    dur   = fitted["primary"]["duration_days"]/period_days
    ing_ph = fitted["primary"]["ingress_minutes"]/(24*60*period_days)

    # 1) Compute how much to shift your fitted mid‐phase to 0.5
    shift = 0.5 - fitted["primary"]["mid_phase"]

    # 2) Apply that same shift to your data
    raw_phase = fold_time(times, period_days, T0_guess)
    idx       = np.argsort(raw_phase)
    phase_shifted = (raw_phase[idx] + shift) % 1
    flux_shifted  = fluxes[idx]

    # 3) Build a model curve but force its mid to 0.5
    dur_ph = fitted["primary"]["duration_days"] / period_days
    ing_ph = fitted["primary"]["ingress_minutes"] / (24*60*period_days)
    phase_model = np.linspace(0, 1, 1000)
    model_flux  = trapezoid_model(
        phase_model,
        0.5,                               # <-- fixed center
        fitted["primary"]["depth"],
        dur_ph,
        ing_ph
    )

    # 4) Shade ingress/egress around 0.5
    half = dur_ph / 2
    i0, i1 = 0.5 - half, 0.5 - half + ing_ph
    e1, e0 = 0.5 + half, 0.5 + half - ing_ph

    # 5) Plot
    plt.figure(figsize=(10,5))
    plt.scatter(phase_shifted, flux_shifted, s=10, alpha=0.6, label="Data")
    plt.plot(phase_model, model_flux, c="red", label="Trapezoid Model")
    plt.axvspan(i0, i1, color="orange", alpha=0.2, label="Ingress Zone")
    plt.axvspan(e0, e1, color="purple", alpha=0.2, label="Egress Zone")
    plt.xlabel("Phase")
    plt.ylabel("Relative Flux")
    plt.title("Trapezoidal Eclipse Fit (centered at 0.5)")
    plt.xlim(0,1)
    plt.ylim(0.65,1.15)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Trapezoidal_Eclipse_Fit_Shifted.png", dpi=300)
    plt.close()

    # 7) print results
    fmin = np.min(fluxes)
    print(f"Eclipse depth = {1-fmin:.3f} flux,  {-2.5*np.log10(fmin):.3f} mag")
    print(calc_ingress_egress(fitted, period_days))


    duration_days = fitted["primary"]["duration_days"]
    duration_minutes = duration_days * 24 * 60 
    duration_hours = duration_days * 24

    print(f"\n Eclipse Duration:")
    print(f" In days:   {duration_days:.5f}")
    print(f" In hours:  {duration_hours:.2f}")
    print(f" In minutes:{duration_minutes:.2f}")

    print("Fitted Trapezoid parameters")
    print(fitted)

    print("Ingress and Egress durations (minutes):")
    print(f" Ingress duration: {fitted['primary']['ingress_minutes']:.2f} minutes")
    print(f" Egress duration: {fitted['primary']['egress_minutes']:.2f} minutes")

    print("\nIngress and Egress times")
    print("Flux min/max:", fluxes.min(), fluxes.max())

    calc_ingress_egress(fitted, period_days)

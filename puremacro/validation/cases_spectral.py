"""Validation cases for the ``spectral`` subsystem (Welch PSD / cross-spectrum /
business-cycle band power).

References are SCIPY (computed live — scipy is a runtime dependency and
pyodide-safe) and INTERNAL identities. No PACKAGE goldens: ``puremacro.spectral``
is a self-contained ``numpy.fft`` implementation of Welch's method, so the
honest, independent yardstick is ``scipy.signal`` configured to the *same*
estimator (symmetric Hann window, constant detrend, density scaling, matched
``noverlap``).

The one convention puremacro and scipy differ on is the one-sided fold:
puremacro returns the *two-sided* spectral density evaluated on the one-sided
grid, whereas ``scipy.signal`` (``return_onesided=True``) folds the
negative-frequency power, doubling every bin except DC and Nyquist. The SCIPY
PSD/CSD cases therefore map puremacro's output through that exact, standard
factor (see :func:`_onesided_factor`); coherence/gain/phase are ratios in which
the factor cancels, so they are compared to scipy directly.

All imports of ``puremacro.spectral`` / ``scipy`` live inside the callables to
keep module import pyodide-light.
"""
from __future__ import annotations

import numpy as np

from ._model import Mechanism, Tol, ValidationCase

# Welch parameters shared by every case and by the scipy reference, so the two
# implementations segment, window, detrend and scale identically.
_N_SEG = 64
_OVERLAP = 0.5
_NOVERLAP = int(round(_N_SEG * (1.0 - _OVERLAP)))  # = 32; scipy noverlap convention
_FS = 1.0


def spectral_demo_data() -> dict:
    """Deterministic demo series for the spectral gallery (seeded → identical).

    ``x`` is a univariate series with persistence (AR(1)) plus a sinusoid sitting
    inside the business-cycle band, so the band-power fraction is non-trivial.
    ``(x2, y2)`` is a bivariate pair where ``y2`` is a lagged, attenuated copy of
    ``x2`` plus idiosyncratic noise, giving partial — strictly interior (0,1) —
    coherence at every frequency.
    """
    rng = np.random.default_rng(20260531)
    T = 400

    # Univariate: AR(1) + a 16-period cycle (inside the 6-32 band) + noise.
    phi = 0.6
    e = rng.standard_normal(T)
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = phi * x[t - 1] + e[t]
    t_idx = np.arange(T)
    x = x + 1.5 * np.sin(2.0 * np.pi * t_idx / 16.0)

    # Bivariate: y2 = 0.7 * x2(t-1) + noise  → partial coherence everywhere.
    x2 = rng.standard_normal(T)
    y2 = 0.7 * np.concatenate([[0.0], x2[:-1]]) + 0.5 * rng.standard_normal(T)

    return {
        "x": x,
        "x2": x2,
        "y2": y2,
        "n_seg": _N_SEG,
        "overlap": _OVERLAP,
        "noverlap": _NOVERLAP,
        "fs": _FS,
    }


def _onesided_factor(n_freq: int) -> np.ndarray:
    """One-sided fold factor for an even-length segment: 2 on every interior bin,
    1 on DC and Nyquist. Maps puremacro's two-sided density to scipy's one-sided.
    """
    fac = np.full(n_freq, 2.0)
    fac[0] = 1.0
    fac[-1] = 1.0  # _N_SEG is even → last rfft bin is the Nyquist bin
    return fac


# --------------------------------------------------------------------------- #
# compute callables (puremacro side)                                          #
# --------------------------------------------------------------------------- #
def _pm_welch_psd() -> dict:
    from puremacro.spectral import welch_psd

    d = spectral_demo_data()
    f, Pxx = welch_psd(d["x"], fs=d["fs"], n_seg=d["n_seg"], overlap=d["overlap"])
    # Fold to the one-sided convention so it lines up with scipy.signal.welch.
    Pxx_onesided = np.asarray(Pxx, dtype=float) * _onesided_factor(len(f))
    return {"f": np.asarray(f, dtype=float), "Pxx": Pxx_onesided}


def _pm_welch_cross_psd() -> dict:
    from puremacro.spectral import welch_cross

    d = spectral_demo_data()
    r = welch_cross(d["x2"], d["y2"], fs=d["fs"], n_seg=d["n_seg"], overlap=d["overlap"])
    fac = _onesided_factor(len(r.f))
    Pxy = np.asarray(r.Pxy)
    return {
        "Pxy_re": Pxy.real * fac,
        "Pxy_im": Pxy.imag * fac,
    }


def _pm_coherence() -> dict:
    from puremacro.spectral import welch_cross

    d = spectral_demo_data()
    r = welch_cross(d["x2"], d["y2"], fs=d["fs"], n_seg=d["n_seg"], overlap=d["overlap"])
    return {"coherence": np.asarray(r.coherence, dtype=float)}


def _pm_band_power() -> dict:
    from puremacro.spectral import business_cycle_band_power

    d = spectral_demo_data()
    bp = business_cycle_band_power(
        d["x"], fs=d["fs"], low_period=6.0, high_period=32.0, n_seg=d["n_seg"]
    )
    return {"band_power": float(bp)}


def _pm_full_band_is_one() -> dict:
    from puremacro.spectral import business_cycle_band_power

    d = spectral_demo_data()
    # high_period=inf → f_low = 1/inf = 0, so the band [0, Nyquist] covers every
    # frequency bin: the fraction must be exactly 1 (variance conservation).
    bp_full = business_cycle_band_power(
        d["x"], fs=d["fs"], low_period=2.0, high_period=float("inf"), n_seg=d["n_seg"]
    )
    return {"full_band_power": float(bp_full)}


def _pm_coherence_bounds() -> dict:
    from puremacro.spectral import welch_cross

    d = spectral_demo_data()
    r = welch_cross(d["x2"], d["y2"], fs=d["fs"], n_seg=d["n_seg"], overlap=d["overlap"])
    coh = np.asarray(r.coherence, dtype=float)
    # Cauchy-Schwarz: 0 <= |Pxy|^2/(Pxx Pyy) <= 1 at every frequency. Encode as two
    # lower bounds (>= 0) checked under QUALITATIVE: coherence and its complement.
    return {
        "coh_min": float(np.min(coh)),
        "one_minus_coh_min": float(np.min(1.0 - coh)),
    }


# --------------------------------------------------------------------------- #
# reference callables (scipy side, computed live)                             #
# --------------------------------------------------------------------------- #
def _ref_scipy_welch() -> dict:
    import scipy.signal as ss

    d = spectral_demo_data()
    win = np.hanning(d["n_seg"])  # symmetric Hann == puremacro _hann == scipy hann(sym=True)
    f, Pxx = ss.welch(
        d["x"],
        fs=d["fs"],
        window=win,
        nperseg=d["n_seg"],
        noverlap=d["noverlap"],
        detrend="constant",
        return_onesided=True,
        scaling="density",
        average="mean",
    )
    return {"f": np.asarray(f, dtype=float), "Pxx": np.asarray(Pxx, dtype=float)}


def _ref_scipy_csd() -> dict:
    import scipy.signal as ss

    d = spectral_demo_data()
    win = np.hanning(d["n_seg"])
    _f, Pxy = ss.csd(
        d["x2"],
        d["y2"],
        fs=d["fs"],
        window=win,
        nperseg=d["n_seg"],
        noverlap=d["noverlap"],
        detrend="constant",
        return_onesided=True,
        scaling="density",
        average="mean",
    )
    Pxy = np.asarray(Pxy)
    return {"Pxy_re": Pxy.real, "Pxy_im": Pxy.imag}


def _ref_scipy_coherence() -> dict:
    import scipy.signal as ss

    d = spectral_demo_data()
    win = np.hanning(d["n_seg"])
    _f, coh = ss.coherence(
        d["x2"],
        d["y2"],
        fs=d["fs"],
        window=win,
        nperseg=d["n_seg"],
        noverlap=d["noverlap"],
        detrend="constant",
    )
    return {"coherence": np.asarray(coh, dtype=float)}


def _ref_scipy_band_power() -> dict:
    """Business-cycle band fraction from scipy's PSD, using the same rectangle
    integration and the same two-sided density puremacro integrates.

    scipy returns the one-sided PSD; dividing the folded interior bins by 2
    recovers the two-sided density on the same grid (a standard, textbook
    one-/two-sided conversion). The band fraction is then a pure scipy-derived
    number, independent of puremacro's code path.
    """
    import scipy.signal as ss

    d = spectral_demo_data()
    win = np.hanning(d["n_seg"])
    f, P_onesided = ss.welch(
        d["x"],
        fs=d["fs"],
        window=win,
        nperseg=d["n_seg"],
        noverlap=d["noverlap"],
        detrend="constant",
        return_onesided=True,
        scaling="density",
        average="mean",
    )
    P_two = np.asarray(P_onesided, dtype=float) / _onesided_factor(len(f))
    df = float(f[1] - f[0])
    low_period, high_period = 6.0, 32.0
    band = (f >= 1.0 / high_period) & (f <= 1.0 / low_period)
    total = float(np.sum(P_two) * df)
    band_power = float(np.sum(P_two[band]) * df) / total
    return {"band_power": band_power}


CASES: list[ValidationCase] = [
    ValidationCase(
        id="spectral.welch_psd_vs_scipy",
        subsystem="spectral",
        title="Welch PSD matches scipy.signal.welch (one-sided, density, Hann)",
        title_es="La DEP de Welch coincide con scipy.signal.welch (unilateral, densidad, Hann)",
        mechanism=Mechanism.SCIPY,
        compute=_pm_welch_psd,
        reference=_ref_scipy_welch,
        tol=Tol.TIGHT,
        citation=(
            "scipy.signal.welch (scipy 1.x): Welch (1967), IEEE Trans. AU-15, 70-73. "
            "Matched window=hann(sym=True), nperseg=64, noverlap=32, "
            "detrend='constant', scaling='density'; puremacro's two-sided density "
            "folded to one-sided (interior bins x2)."
        ),
        notes="Both segment, window, detrend and scale identically; only the one-sided fold differs.",
    ),
    ValidationCase(
        id="spectral.welch_cross_vs_scipy_csd",
        subsystem="spectral",
        title="Welch cross-spectrum Pxy matches scipy.signal.csd",
        title_es="El espectro cruzado Pxy de Welch coincide con scipy.signal.csd",
        mechanism=Mechanism.SCIPY,
        compute=_pm_welch_cross_psd,
        reference=_ref_scipy_csd,
        tol=Tol.TIGHT,
        citation=(
            "scipy.signal.csd (scipy 1.x), conjugation convention Pxy = E[conj(X)Y]. "
            "Matched window=hann(sym=True), nperseg=64, noverlap=32, "
            "detrend='constant', scaling='density'; folded to one-sided."
        ),
        notes="Real and imaginary parts of the cross-spectrum compared separately.",
    ),
    ValidationCase(
        id="spectral.coherence_vs_scipy",
        subsystem="spectral",
        title="Magnitude-squared coherence matches scipy.signal.coherence",
        title_es="La coherencia cuadrática coincide con scipy.signal.coherence",
        mechanism=Mechanism.SCIPY,
        compute=_pm_coherence,
        reference=_ref_scipy_coherence,
        tol=Tol.TIGHT,
        citation=(
            "scipy.signal.coherence (scipy 1.x): |Pxy|^2 / (Pxx Pyy). "
            "Matched window=hann(sym=True), nperseg=64, noverlap=32, "
            "detrend='constant'. Being a ratio, the one-sided fold factor cancels."
        ),
        notes="Coherence is convention-free (the x2 fold cancels in numerator and denominator).",
    ),
    ValidationCase(
        id="spectral.band_power_vs_scipy",
        subsystem="spectral",
        title="Business-cycle (6-32) band-power fraction matches scipy-PSD integration",
        title_es="La fracción de potencia en la banda de ciclo económico (6-32) coincide con la integración de la DEP de scipy",
        mechanism=Mechanism.SCIPY,
        compute=_pm_band_power,
        reference=_ref_scipy_band_power,
        tol=Tol.TIGHT,
        citation=(
            "Burns & Mitchell (1946) business-cycle band of 6-32 quarters; "
            "reference fraction integrates the scipy.signal.welch PSD over the band "
            "with the same rectangle rule, on the two-sided density."
        ),
        notes="Direct scipy-PSD band integration; same Welch config as the estimator.",
    ),
    ValidationCase(
        id="spectral.full_band_power_is_one",
        subsystem="spectral",
        title="Band power over the full frequency range equals 1 (variance conservation)",
        title_es="La potencia de banda sobre todo el rango de frecuencias es igual a 1 (conservación de la varianza)",
        mechanism=Mechanism.INTERNAL,
        compute=_pm_full_band_is_one,
        reference=lambda: {"full_band_power": 1.0},
        tol=Tol.EXACT,
        citation=(
            "Partition-of-unity / Parseval identity: integrating the PSD over the "
            "entire frequency support recovers the total variance, so the fraction "
            "is exactly 1 (Lütkepohl 2005, spectral representation)."
        ),
        notes="high_period=inf sets f_low=0, covering every bin including DC and Nyquist.",
    ),
    ValidationCase(
        id="spectral.coherence_in_unit_interval",
        subsystem="spectral",
        title="Magnitude-squared coherence lies in [0, 1] (Cauchy-Schwarz)",
        title_es="La coherencia cuadrática pertenece a [0, 1] (Cauchy-Schwarz)",
        mechanism=Mechanism.INTERNAL,
        compute=_pm_coherence_bounds,
        reference=lambda: {"coh_min": 0.0, "one_minus_coh_min": 0.0},
        tol=Tol.QUALITATIVE,
        citation=(
            "Cauchy-Schwarz bound on the cross-spectrum: 0 <= |Pxy|^2/(Pxx Pyy) <= 1 "
            "at every frequency (Brockwell & Davis 1991, §11.6)."
        ),
        notes="Both the coherence floor and 1 - coherence floor must be >= 0.",
    ),
]

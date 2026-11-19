"""Quality indicators for the adopted Voigt curves, on the original scan grid."""
import numpy as np
import pandas as pd
from compare_asymmetry import harmonic
from survey import inputs


def assess(f, residual, center, width, height, n_parameters):
    lag = float(np.corrcoef(residual[:-1], residual[1:])[0, 1])
    noise = np.sqrt(residual @ residual / (len(f) - n_parameters))
    snr = height / (2 * noise)  # background-subtracted, one-sided peak amplitude
    span = min(center - f.min(), f.max() - center) / 1000
    return dict(residual_lag1=lag, signal_to_noise=snr, baseline_span_kHz=span,
                weak_signal=snr < 5, correlated_residuals=abs(lag) > .3,
                short_baseline_range=span < width)


def evaluate(table, models):
    paths = {item[1].name: item[1] for item in inputs()}
    fits = models.set_index(["filename", "model"], verify_integrity=True)
    rows = []
    for row in table.itertuples():
        model = "Voigt" if row.task == 2 else "finite_Voigt"
        assert row.model == model
        fit = fits.loc[(row.file, model)]
        f, y = np.loadtxt(paths[row.file]).T
        x = (f - fit.reference_Hz) / 1000
        u = (f - fit.f0_Hz) / 1000
        shape = harmonic(u, fit.gamma_kHz, fit.sigma_kHz, fit.excursion_kHz, nodes=256).real
        residual = y - (fit.amplitude * shape + fit.background + fit.slope_per_kHz*x)
        np.testing.assert_allclose(np.sqrt(np.mean(residual**2)), row.rms, rtol=1e-5)
        rows.append(assess(f, residual, fit.f0_Hz, row.fwhm_kHz, row.height,
                           6 if row.task == 2 else 7))
    result = table.drop(columns=["baseline_flag"], errors="ignore").copy()
    for key in rows[0]:
        result[key] = [r[key] for r in rows]
    return result

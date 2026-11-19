"""Independent line-shape comparison on four ascending CW-NMR sweeps."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import lfilter
from scipy.special import wofz

PACKAGE = Path(__file__).resolve().parents[1]
OUT = PACKAGE / "作業用" / "比較"
OUT.mkdir(parents=True, exist_ok=True)
NMR = PACKAGE / "06_入力"
TASK1 = NMR / "課題1_共鳴信号"
TASK3 = NMR / "課題3_変調条件と追加測定"
RESULTS = TASK3 / "解析III" / "results"


def harmonic(offset, gamma, sigma, excursion, nodes=64):
    # Real and imaginary parts share the same normalization and frequency axis.
    scale = np.sqrt(2) * sigma
    if excursion == 0:
        z = (offset + 1j * gamma) / scale
        return (-2*z*wofz(z) + 2j/np.sqrt(np.pi)) / (scale*sigma)
    cosine = np.cos((np.arange(nodes) + .5) * np.pi / nodes)
    z = (offset[:, None] + excursion*cosine + 1j*gamma) / scale
    return 2*np.mean(wofz(z)*cosine, axis=1) / (excursion*sigma)


def fit(x, y, mask, start, excursion, mode, memory):
    def solve(q, nodes=64):
        center, gamma, sigma = q
        h = harmonic(x-center, gamma, sigma, excursion, nodes)
        columns = [h.real]
        if mode == "mixed":
            columns.append(h.imag)
        columns.extend([np.ones_like(x), x])
        matrix = np.column_stack(columns)
        if mode == "lag":
            matrix, _ = lfilter([1-memory], [1., -memory], matrix, axis=0,
                                zi=memory*matrix[:1])
        beta = np.linalg.lstsq(matrix[mask], y[mask], rcond=None)[0]
        return matrix @ beta, beta

    result = least_squares(lambda q: (solve(q)[0]-y)[mask], start,
                           bounds=([-4, .02, .025], [4, 10, 10]),
                           ftol=1e-10, xtol=1e-10, gtol=1e-10)
    if not result.success:
        raise RuntimeError(result.message)
    predicted, beta = solve(result.x, nodes=128)
    np.testing.assert_allclose(solve(result.x)[0], predicted, atol=1e-6)
    return result.x, predicted, beta


def scans():
    ref = 16010622.762131764
    yield "Task 1", TASK1/"元データ"/"20261020_KITtest01.dat", ref, [.484, 1.035], 0., .5
    fits = json.loads((RESULTS/"model_comparison"/"parameters.json").read_text())
    table = pd.read_csv(RESULTS/"fit_results.csv").set_index("filename")
    for v in [.5, 1.3, 4.5]:
        filename = f"20261104_v{v:.1f}_f280_01.dat"
        offset, gamma, sigma, excursion = fits[filename]["finite_voigt"]["q"]
        ref = table.loc[filename, "f0_Hz"] + 1000*offset
        yield f"Task 3, {v} V", TASK3/"20261104"/filename, ref, [gamma,sigma], excursion, 1.


def rms(values):
    return np.sqrt(np.mean(values**2))


def main():
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.4), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    rows = []
    validation = []
    for idx, (label, path, ref, widths, excursion, wait) in enumerate(scans()):
        f, y = np.loadtxt(path).T
        assert np.all(np.diff(f) > 0)
        x = (f-ref)/1000
        core = abs(x) <= 6
        # Hold out consecutive 2-kHz intervals, so adjacent samples inside each
        # interval do not leak into training. The excursion is conditioned on
        # the earlier whole-scan estimate for all Task 3 models.
        for mode in ["symmetric", "mixed", "lag"]:
            held_residuals = []
            for left in range(-6, 6, 2):
                held = (x >= left) & ((x <= 6) if left == 4 else (x < left+2))
                if not np.any(held):
                    continue
                _, pred, _ = fit(x, y, ~held, [0, *widths], excursion, mode, np.exp(-wait/.2))
                held_residuals.extend((y-pred)[held])
            assert len(held_residuals) == np.count_nonzero(core)
            validation.append(dict(scan=label, model=mode, n_core=len(held_residuals),
                                   held_interval_rms=rms(np.array(held_residuals))))
        for window in [np.inf, 10]:
            mask = abs(x) <= window
            for mode in ["symmetric", "mixed", "lag"]:
                q, pred, beta = fit(x, y, mask, [0, *widths], excursion, mode, np.exp(-wait/.2))
                res = y-pred
                rows.append(dict(scan=label, file=path.name, window_kHz=window, model=mode,
                                 n=mask.sum(), excursion_kHz=excursion, f0_Hz=ref+1000*q[0],
                                 gamma_kHz=q[1], sigma_kHz=q[2], dispersion_over_absorption=(
                                     beta[1]/beta[0] if mode=="mixed" else 0),
                                 rms=rms(res[mask]), core6_rms=rms(res[core])))
                if np.isinf(window) and mode in ["symmetric","mixed"]:
                    top, bottom = axes[:, idx]
                    style, color = ("--","#155091") if mode=="symmetric" else ("-","#bd411d")
                    keep = abs(x) <= 9
                    if mode=="symmetric":
                        top.plot(x[keep], y[keep], ".", color=".42", ms=3, label="Measured")
                    top.plot(x[keep], pred[keep], style, color=color, lw=1.5, label=mode.title())
                    bottom.plot(x[keep], res[keep], style, color=color, lw=.95, marker=".", ms=2)
        top.set_title(label)
        for ax in (top, bottom):
            ax.grid(alpha=.2)
            ax.axhline(0, lw=.6, color=".5")
    axes[0,0].set_ylabel("Recorded signal")
    axes[1,0].set_ylabel("Measured minus fit")
    axes[0,0].legend(fontsize=8)
    fig.supxlabel("Frequency offset from symmetric fit center (kHz)")
    fig.tight_layout()
    fig.savefig(OUT/"waveforms.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT/"comparison.csv", index=False)
    pd.DataFrame(validation).to_csv(OUT/"held_intervals.csv", index=False)


if __name__ == "__main__":
    main()

"""Rebuild report figures from adopted fit results, without refitting data.

Run from any directory: python -B plot_report_figures.py
Task 1: compare saved Lorentz and Voigt fits and their ordinary residuals.
Task 2: plot retained positions using measurement-date markers.
Task 3: show only the baseline and finite-modulation Voigt comparison.
"""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
NMR = HERE.parent
TASK1 = NMR / "課題1_共鳴信号" / "解析I"
TASK2 = NMR / "課題2_位置依存" / "解析II"
TASK3 = NMR / "課題3_変調条件と追加測定" / "解析III"
sys.path.insert(0, str(TASK1))
sys.path.insert(0, str(TASK2))
from nmr_joint_noise import modulated_voigt
from nmr_physical_models import lorentz_derivative
from position_plot_selection import position_fit_display

OUT = HERE / "figures"
BLUE, ORANGE = "#245a9b", "#c55a20"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def task1():
    stem = "20261020_KITtest01"
    f, y = np.loadtxt(TASK1 / (stem + ".dat")).T
    saved = pd.read_csv(TASK1 / (stem + "_joint_noise_residuals.csv"))
    summary = read_json(TASK1 / (stem + "_joint_noise_summary.json"))
    physical = read_json(TASK1 / (stem + "_physical_models.json"))
    ref = summary["reference_Hz"]
    assert ref == physical["assumptions"]["frequency_reference_Hz"]
    np.testing.assert_allclose(f, saved.frequency_Hz, rtol=0, atol=1e-8)
    np.testing.assert_allclose(y, saved.signal, rtol=0, atol=1e-12)
    x = (f - ref) / 1000
    p = summary["shape_parameters_kHz"]
    q = [p[k] for k in ("center", "gamma", "sigma", "modulation_peak")]
    b = summary["linear_parameters"]
    beta = [b[k] for k in ("amplitude", "offset", "slope_per_kHz")]
    assert q[3] == 0, "Update the Voigt legend if a finite modulation is adopted."
    lp = physical["lorentz"]
    lc, lg = lp["shape_parameters"]
    la, lb, ls = lp["linear_parameters"]

    def voigt(t):
        return beta[0] * modulated_voigt(t-q[0], *q[1:], nodes=128) + beta[1] + beta[2]*t

    def lorentz(t):
        return la * lorentz_derivative(t-lc, lg) + lb + ls*t

    np.testing.assert_allclose(voigt(x), saved.mean_fit, rtol=0, atol=1e-6)
    core = abs(x) < 5
    assert len(y) == 1250 and core.sum() == 75
    expected = {"Voigt derivative": .06466000731347785,
                "Lorentz derivative": .10165762968510776}
    mask = abs(x) < 7
    grid = np.linspace(-7, 7, 1600)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    axes[0].scatter(x[mask], y[mask], s=17, c="black", alpha=.6,
                    label="Measured", zorder=3)
    for fn, label, color, style in [(voigt, "Voigt derivative", BLUE, "-"),
                                    (lorentz, "Lorentz derivative", ORANGE, "--")]:
        residual = y - fn(x)
        rms = np.sqrt(np.mean(residual[core]**2))
        assert np.isclose(rms, expected[label], rtol=1e-6)
        axes[0].plot(grid, fn(grid), style, color=color, lw=1.7, label=label)
        axes[1].plot(x[mask], residual[mask], marker=".", ls=style,
                     color=color, ms=4, lw=.8, label=label)
        print(f"Task 1 {label}: core RMS={rms:.7f}")
    axes[0].set_ylabel("Lock-in output (recorded units)")
    axes[1].set_ylabel("Residual\n(recorded units)")
    axes[1].set_xlabel(f"Frequency offset from {ref/1e6:.6f} MHz (kHz)")
    axes[0].legend(fontsize=10)
    axes[1].legend(fontsize=9, ncol=2)
    for ax in axes:
        ax.axhline(0, c="black", lw=.6)
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT / "task1_fit_residuals.png", dpi=200)
    plt.close(fig)


def task2():
    frame = pd.read_csv(TASK2 / "results_with_additions" / "fit_results.csv")
    display = frame.apply(position_fit_display, axis=1)
    estimates = frame[display == "estimate"].copy()
    withheld = frame[display == "withheld"]
    reference = frame[display == "reference"]
    assert not frame[display != "excluded"].duplicated(["x", "z"]).any()
    estimates["field_mT"] = estimates.f0_Hz / 42.5759e6 * 1000
    estimates["width_kHz"] = estimates.fwhm_Hz / 1000
    estimates["strength"] = 9 * abs(estimates.L0) / (4 * np.sqrt(3) * estimates.hwhm_Hz)
    markers = {20261021: ("o", "Oct 21"), 20261027: ("s", "Oct 27"),
               20261028: ("^", "Oct 28"), 20261104: ("D", "Nov 4"),
               20261110: ("P", "Nov 10")}
    assert set(estimates.date.astype(int)) == set(markers)
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.5))
    metrics = [("field_mT", "(a) Inferred magnetic field", "Field (mT)", "viridis"),
               ("width_kHz", "(b) Absorption FWHM", "Width (kHz)", "cividis"),
               ("strength", "(c) Signal strength", "Peak-to-peak (recorded units)", "magma")]
    for ax, (key, title, label, cmap) in zip(axes, metrics):
        norm = plt.Normalize(estimates[key].min(), estimates[key].max())
        for date, (marker, _) in markers.items():
            part = estimates[estimates.date == date]
            ax.scatter(part.x, part.z, c=part[key], cmap=cmap, norm=norm,
                       marker=marker, s=105, edgecolors="#454545", linewidths=.45,
                       zorder=3)
        ax.scatter(withheld.x, withheld.z, marker="x", c="#666666", s=95, linewidths=1.6, zorder=4)
        if key == "field_mT":
            ax.scatter(reference.x, reference.z, marker="o", facecolors="none",
                       edgecolors="#333333", s=105, linewidths=1.4, zorder=4)
        ax.set(xlabel="x (cm)", ylabel="z (cm)", title=title,
               xlim=(2.15, 6.85), ylim=(2.15, 6.85))
        ax.set_xticks(np.arange(2.5, 7, 1))
        ax.set_yticks(np.arange(2.5, 7, 1))
        ax.set_aspect("equal")
        ax.grid(alpha=.25)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                     orientation="horizontal", pad=.19, fraction=.05, label=label)
    handles = [Line2D([], [], marker=m, ls="", color="#454545", ms=8, label=l)
               for m, l in markers.values()]
    handles.extend([Line2D([], [], marker="o", markerfacecolor="none", ls="", color="#333333",
                           ms=8, label="Field reference"),
                    Line2D([], [], marker="x", ls="", color="#666666", ms=8, label="Estimate withheld")])
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .93), w_pad=3)
    fig.savefig(OUT / "task2_position.png", dpi=200)
    plt.close(fig)
    print(f"Task 2: {len(estimates)} colored positions, {len(reference)} field reference, "
          f"{len(withheld)} withheld.")


def task3():
    table = pd.read_csv(TASK3 / "results" / "fit_results.csv")
    rows = table[(table.voltage_V == 4.5) & (table.modulation_Hz == 280)]
    assert len(rows) == 1
    row = rows.iloc[0]
    f, y = np.loadtxt(TASK3.parent / "20261104" / row.filename).T
    details = read_json(TASK3 / "results" / "model_comparison" / "parameters.json")
    fit = details[row.filename]["finite_voigt"]
    q, b = fit["q"], fit["beta"]
    ref = row.f0_Hz
    x = (f-ref)/1000

    def lorentz(t):
        u = (ref + 1000*t) - row.f0_Hz
        g = row.hwhm_Hz
        return (-2*row.L0*g*g*u/(g*g+u*u)**2 + row.baseline +
                row.baseline_slope_per_Hz*(ref+1000*t-row.f_ref_Hz))

    def finite(t):
        return b[0]*modulated_voigt(t-q[0], *q[1:], nodes=2*fit["nodes"]) + b[1] + b[2]*t

    np.testing.assert_allclose(np.sum((y-finite(x))**2), fit["sse"], rtol=1e-5)
    np.testing.assert_allclose(np.sqrt(np.mean((y-lorentz(x))**2)), row.residual_rms, rtol=1e-6)
    radius = max(6., 3*row.hwhm_Hz/1000)
    grid = np.linspace(-radius, radius, 1600)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(x, y, ".", c="black", ms=5, label="Measured")
    for fn, label, color, style in [(lorentz, "Lorentz derivative", ORANGE, "--"),
                                    (finite, "Finite-modulation Voigt", BLUE, "-")]:
        axes[0].plot(grid, fn(grid), style, c=color, lw=1.6, label=label)
        axes[1].plot(x, y-fn(x), marker=".", ls=style, c=color, lw=.8, ms=4)
    axes[0].set_ylabel("Recorded signal")
    axes[1].set_ylabel("Measured - fit")
    axes[1].set_xlabel(f"Frequency offset from {ref/1e6:.6f} MHz (kHz)")
    axes[0].legend(fontsize=10)
    axes[0].set_title("4.5 V, 280 Hz")
    for ax in axes:
        ax.set_xlim(-radius, radius)
        ax.axhline(0, c="black", lw=.6)
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT / "task3_finite_modulation.png", dpi=200)
    plt.close(fig)
    print("Task 3: two saved models, same measured points; no refit.")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 11})
    task1()
    task2()
    task3()

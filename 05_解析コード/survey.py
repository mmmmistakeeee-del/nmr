"""Compare dispersion mixing on the accepted CW-NMR scans, in a separate directory."""
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares

from compare_asymmetry import harmonic, TASK1, TASK3, RESULTS, NMR, OUT

T2 = NMR/"課題2_位置依存"
sys.path.insert(0, str(T2/"解析II"))
from position_plot_selection import position_fit_display


def read_table(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def inputs():
    yield (1, TASK1/"元データ"/"20261020_KITtest01.dat", 16010622.762, 1.7, "", "", "", "", "")
    paths = [T2/"実験II", T2/"測定記録"/"20261104", T2/"測定記録"/"20261110"]
    for row in read_table(T2/"解析II"/"results_with_additions"/"fit_results.csv"):
        if position_fit_display(row) != "estimate":
            continue
        matches = [folder/row["filename"] for folder in paths if (folder/row["filename"]).is_file()]
        assert len(matches)==1 and row["segment"]=="1" and row["sweep"]=="up"
        yield (2, matches[0], float(row["f0_Hz"]), float(row["hwhm_Hz"])/1000,
               row["x"], row["z"], "", "", row["warning"])
    for row in read_table(RESULTS/"fit_results.csv"):
        if not row["voltage_V"] or row["position_series"]=="True":
            continue
        hz = float(row["modulation_Hz"])
        if hz in (20, 60):
            continue
        yield (3, TASK3/"20261104"/row["filename"], float(row["f0_Hz"]),
               float(row["hwhm_Hz"])/1000, "", "", row["voltage_V"], hz, row["warning"])


def fit(x, y, train, mixed, finite, width, seeds):
    def solve(q, nodes):
        center, gamma, sigma = q[:3]
        h = harmonic(x-center, gamma, sigma, q[3] if finite else 0, nodes=nodes)
        cols = [h.real, h.imag] if mixed else [h.real]
        m = np.column_stack(cols + [np.ones_like(x), x])
        beta = np.linalg.lstsq(m[train], y[train], rcond=None)[0]
        return m@beta-y, beta

    # The preliminary resonance locates the search region only. All shape and
    # background parameters are refitted from the training samples.
    extent = max(4., 1.5*width)
    lower = [max(x.min(),-extent), .02, .04] + ([.001] if finite else [])
    upper = [min(x.max(),extent), max(10.,4*width), max(10.,4*width)] + ([max(10.,4*width)] if finite else [])
    nodes = 32
    while True:
        best = None
        attempts = []
        for seed in seeds:
            r = least_squares(lambda q: solve(q,nodes)[0][train],
                              np.clip(seed,np.array(lower)+1e-6,np.array(upper)-1e-6),
                              bounds=(lower,upper), max_nfev=180, ftol=2e-8, xtol=2e-8)
            attempts.append(r)
            if r.success and (best is None or r.cost < best.cost):
                best = r
        if best is None:
            candidate = min(attempts,key=lambda attempt: attempt.cost)
            best = least_squares(lambda q: solve(q,nodes)[0][train],candidate.x,
                                 bounds=(lower,upper),x_scale="jac",max_nfev=1400,
                                 ftol=1e-7,xtol=1e-7)
            if not best.success:
                raise RuntimeError("No converged fit")
        err, beta = solve(best.x,2*nodes)
        if not finite or np.max(abs(err-solve(best.x,nodes)[0])) < 2e-5 * np.ptp(y):
            return best.x, err, beta
        if nodes >= 256:
            raise RuntimeError("Harmonic integration failed to converge")
        nodes *= 2
        seeds = [best.x, *seeds]


def analyze(item, finite_position=False):
    task,path,ref,width,px,pz,volts,hz,warning = item
    f,y = np.loadtxt(path).T
    assert np.isfinite(f+y).all() and (np.diff(f)>=0).all()
    x = (f-ref)/1000
    finite = task==3 or (task==2 and finite_position)
    seeds = [[0,.4*width,.55*width], [0,.95*width,.12*width]]
    if finite:
        seeds = [seeds[0]+[.85*width], seeds[1]+[.3*width],
                 [0,.18*width,.18*width,1.35*width]]
    # Locate fold edges at the same symmetric full-fit center for both models;
    # the measured voltages inside each held interval are excluded from fitting.
    qall,_,_ = fit(x,y,np.ones(len(x),bool),False,finite,width,seeds)
    offset = x-qall[0]
    result = dict(task=task,filename=path.name,x_cm=px,z_cm=pz,voltage_V=volts,modulation_Hz=hz,
                  baseline_warning=warning,n=len(x))
    for mixed, name in [(False,"symmetric"),(True,"mixed")]:
        q,_,beta = fit(x,y,np.ones(len(x),bool),mixed,finite,width,seeds)
        if mixed:
            result["dispersion_over_absorption"] = beta[1]/beta[0]
            result["center_change_kHz"] = q[0]-qall[0]
        errors = []
        block_mse = []
        counts = []
        for left in range(-6,6,2):
            held = (offset>=left)&((offset<=6) if left==4 else (offset<left+2))
            counts.append(held.sum())
            if not held.any():
                continue
            _,err,_ = fit(x,y,~held,mixed,finite,width,seeds)
            errors.extend(err[held])
            block_mse.append(np.mean(err[held]**2))
        result["n_test"] = sum(counts)
        result["n_blocks"] = len(block_mse)
        result["min_points_per_block"] = min(counts)
        result["rms_"+name] = np.sqrt(np.mean(np.square(errors)))
        if not mixed:
            base_mse = block_mse
        else:
            result["blocks_improved"] = sum(np.array(block_mse) < base_mse)
    result["reduction_pct"] = 100*(1-result["rms_mixed"]/result["rms_symmetric"])
    return result


def analyze_position(item):
    try:
        return analyze(item,finite_position=True)
    except RuntimeError as error:
        task,path,_,_,px,pz,_,_,warning = item
        return dict(task=task,filename=path.name,x_cm=px,z_cm=pz,baseline_warning=warning,
                    fit_issue=str(error))


def plot_survey():
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    import pandas as pd
    plt.rcParams.update({"font.family":["Yu Gothic","DejaVu Sans"],"font.size":10})
    data = pd.read_csv(OUT/"survey.csv")
    positions = pd.read_csv(OUT/"position_finite.csv")
    mod = data[data.task==3]
    fig, axes = plt.subplots(1,3,figsize=(13.2,4.5),layout="constrained")
    ax = axes[0]
    sc = ax.scatter(positions.x_cm,positions.z_cm,c=positions.reduction_pct,
                    cmap="RdBu",norm=TwoSlopeNorm(vmin=-50,vcenter=0,vmax=50),
                    s=62,edgecolor=".42",linewidth=.5)
    ax.set(aspect="equal",xlabel="x (cm)",ylabel="z (cm)",title="課題2　位置による違い")
    fig.colorbar(sc,ax=ax,location="bottom",shrink=.86,pad=.15,extend="both",
                 label="予測誤差の減少率 (%)")
    for ax,group,xcol,title,xlabel in [
        (axes[1],mod[mod.modulation_Hz==280],"voltage_V","課題3　280 Hz 固定","変調電圧 (V)"),
        (axes[2],mod[mod.voltage_V==1.3],"modulation_Hz","課題3　1.3 V 固定","変調周波数 (Hz)")]:
        ax.axhline(0,color=".35",lw=.9)
        ax.scatter(group[xcol],group.reduction_pct,
                   c=np.where(group.reduction_pct>0,"#185b85","#ac483a"),s=38)
        ax.set(title=title,xlabel=xlabel,ylabel="予測誤差の減少率 (%)")
        ax.grid(axis="y",alpha=.24)
    axes[1].set_ylim(-110,60)
    axes[2].set_ylim(-190,65)
    ref180 = mod[(mod.voltage_V==1.3)&(mod.modulation_Hz==180)].iloc[0]
    axes[2].annotate("180 Hz　背景の影響あり",(180,ref180.reduction_pct),
                     xytext=(8,0),textcoords="offset points",va="center",fontsize=8)
    fig.savefig(OUT/"survey_summary.png",dpi=160)
    plt.close(fig)


if __name__=="__main__":
    import argparse
    from collections import Counter
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finite-position",action="store_true",
                        help="Check Task 2 with a free finite modulation amplitude")
    parser.add_argument("--plot-only",action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        plot_survey()
        sys.exit()
    survey = list(inputs())
    assert Counter(k[0] for k in survey)=={1:1,2:78,3:27}
    if args.finite_position:
        survey = [item for item in survey if item[0]==2]
    with ProcessPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(analyze_position if args.finite_position else analyze, survey))
    filename = "position_finite.csv" if args.finite_position else "survey.csv"
    with (OUT/filename).open("w",encoding="utf-8-sig",newline="") as handle:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

"""Render the adopted report fits without optimizing any parameters."""
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Callable
import json
import sys

import numpy as np
import pandas as pd
from compare_asymmetry import PACKAGE, TASK1, harmonic, plt
from survey import inputs
from adopted_diagnostics import assess

TABLES = PACKAGE / "04_解析結果"
DEST = PACKAGE / "03_図表" / "個別フィット"
TASK1_ANALYSIS = TASK1 / "解析I"
sys.path.insert(0, str(TASK1_ANALYSIS))
from nmr_joint_noise import modulated_voigt


@dataclass
class Scan:
    path: Path
    folder: Path
    title: str
    model: str
    center: float  # Hz
    width: float   # absorption FWHM, kHz
    rms: float
    predict: Callable  # accepts kHz relative to center
    note: str = ""


def first_scan(path):
    summary = json.loads((TASK1_ANALYSIS / (path.stem + "_joint_noise_summary.json")).read_text())
    shape = summary["shape_parameters_kHz"]
    beta = summary["linear_parameters"]
    params = pd.read_csv(TABLES / "課題1_推定値と不確かさ.csv").set_index("識別子")["元データの推定値"]
    assert shape["modulation_peak"] == 0
    center = summary["reference_Hz"] + 1000 * shape["center"]
    def predict(u):
        line = modulated_voigt(u, shape["gamma"], shape["sigma"], 0)
        return beta["amplitude"] * line + beta["offset"] + beta["slope_per_kHz"] * (u + shape["center"])
    f, y = np.loadtxt(path).T
    saved = pd.read_csv(TASK1_ANALYSIS / (path.stem + "_joint_noise_residuals.csv"))
    np.testing.assert_allclose(f, saved.frequency_Hz, rtol=0, atol=1e-7)
    np.testing.assert_allclose(y, saved.signal, rtol=0, atol=1e-10)
    np.testing.assert_allclose(predict((f-center)/1000), saved.mean_fit, rtol=0, atol=1e-7)
    residual = y - saved.mean_fit.to_numpy()
    diag = assess(f, residual, center, params["absorption_Voigt_FWHM_kHz"],
                  np.ptp(saved.fitted_resonance), 6)
    core = np.abs(f - summary["reference_Hz"]) < 5000
    core_residual = residual[core]
    core_lag = np.corrcoef(core_residual[:-1], core_residual[1:])[0,1]
    note = f"全点の隣接相関 {diag['residual_lag1']:.2f} ／ 共鳴付近（±5 kHz）の残差相関 {core_lag:.2f}"
    return Scan(path, Path("課題1_共鳴信号"), "課題1　共鳴信号", "微分Voigt型",
                center, params["absorption_Voigt_FWHM_kHz"], summary["RMS"], predict, note)


def remaining_scan(task, path, row, note):
    if task == 2:
        folder = Path("課題2_位置依存", path.name[:8])
        title = f"課題2　位置 (x, z) = ({row.x_cm:g}, {row.z_cm:g}) cm"
        model = "微分Voigt型"
    else:
        series = "両系列共通" if row.voltage_V == 1.3 and row.modulation_Hz == 280 else (
            "電圧依存_280Hz" if row.modulation_Hz == 280 else "周波数依存_1.3V")
        folder = Path("課題3_変調条件", series)
        title = f"課題3　変調電圧 {row.voltage_V:g} V　変調周波数 {row.modulation_Hz:g} Hz"
        model = "有限変調Voigt型"
    offset = (row.f0_Hz - row.reference_Hz) / 1000
    def predict(u):
        line = harmonic(u, row.gamma_kHz, row.sigma_kHz, row.excursion_kHz, nodes=256).real
        return row.amplitude * line + row.background + row.slope_per_kHz * (u + offset)
    return Scan(path, folder, title, model, row.f0_Hz, row.absorption_fwhm_kHz, row.full_rms, predict, note)


def draw(scan):
    f, y = np.loadtxt(scan.path).T
    u = (f - scan.center) / 1000
    residual = y - scan.predict(u)
    np.testing.assert_allclose(np.sqrt(np.mean(residual**2)), scan.rms, rtol=1e-5,
                               err_msg=scan.path.name)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 5.9), sharex="col", height_ratios=[2, 1])
    fig.subplots_adjust(left=.09, right=.98, top=.78, bottom=.18, hspace=.12, wspace=.22)
    fig.suptitle(scan.title, fontsize=17, y=.975)
    fig.text(.5,.89, f"{scan.model}　　$f_0$ = {scan.center/1e6:.6f} MHz　　吸収線の半値全幅 {scan.width:.3f} kHz",
             ha="center")
    radius = max(4, 2.3 * scan.width)
    for col,(lo,hi) in enumerate([(u.min(),u.max()), (max(-radius,u.min()), min(radius,u.max()))]):
        top, bottom = axes[:,col]
        mask = (u>=lo) & (u<=hi)
        grid = np.linspace(lo,hi,1200)
        top.plot(u[mask],y[mask],".",color="#596975",ms=3.4,label="測定値")
        top.plot(grid,scan.predict(grid),color="#bb4a23",lw=1.6,label="採用フィット")
        top.set_title(["全測定範囲","共鳴付近"][col], fontsize=11)
        bottom.axhline(0,color="#79868b",lw=.8)
        bottom.plot(u[mask],residual[mask],".",color="#215a76",ms=3.4)
        bottom.set_xlabel("共鳴周波数からの差 (kHz)")
        for ax in (top,bottom):
            ax.set_xlim(lo,hi)
            ax.grid(alpha=.15)
    axes[0,0].legend(loc="best",frameon=False,fontsize=8)
    axes[0,0].set_ylabel("ロックイン出力（記録単位）")
    axes[1,0].set_ylabel("残差（記録単位）")
    fig.text(.09,.063, f"{scan.path.name}    |    全点残差 RMS {scan.rms:.4f}", fontsize=9)
    if scan.note:
        fig.text(.09,.023, scan.note, fontsize=9, color="#754520")
    target = DEST / scan.folder / scan.path.with_suffix(".png").name
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary = target.with_suffix(".tmp.png")
    fig.savefig(temporary,dpi=130)
    temporary.replace(target)
    plt.close(fig)
    return target


def main():
    plt.rcParams.update({"font.family":["Yu Gothic","DejaVu Sans"],"axes.spines.top":False,
                         "axes.spines.right":False,"axes.unicode_minus":False})
    models = pd.read_csv(TABLES / "4モデルの比較_全420件.csv").set_index(["filename","model"])
    adopted = {task: pd.read_csv(next(TABLES.glob(f"課題{task}_主解析表.csv"))).set_index("ファイル名")
               for task in (2,3)}
    catalog = []
    seen = {2:set(),3:set()}
    for task,path,*_ in inputs():
        if task == 1:
            scan = first_scan(path)
        else:
            primary = adopted[task].loc[path.name]
            row = models.loc[(path.name,"Voigt" if task==2 else "finite_Voigt")]
            np.testing.assert_allclose(row.f0_Hz,primary["共鳴周波数 (MHz)"]*1e6,rtol=0,atol=1e-5)
            np.testing.assert_allclose(row.absorption_fwhm_kHz,primary["変調前の吸収線半値全幅 (kHz)"])
            fit_note = primary["採用モデルの注意"]
            obs = primary["測定上の注記"]
            note = f"全点の隣接相関 {primary['全点残差の隣接相関']:.2f}　S/N {primary['片側ピークのS/N']:.1f}"
            if pd.notna(fit_note):
                note += "　｜　" + fit_note
            if pd.notna(obs):
                note += "　｜　測定上の注記：" + obs
            scan = remaining_scan(task,path,row,note if pd.notna(note) else "")
            seen[task].add(path.name)
        catalog.append((scan,draw(scan).relative_to(DEST).as_posix()))
    assert all(seen[t] == set(adopted[t].index) for t in adopted)
    assert len(set(p for _,p in catalog)) == len(catalog) == 106
    lines = ["# 採用フィットの波形と残差","",
             "課題1は1波形、課題2は78座標、課題3は27条件。各図の左は全点、右は共鳴付近。下段は測定値からフィット値を引いた残差。線幅は変調前の吸収線の半値全幅。","",
             "課題3の1.3 V・280 Hzは両系列共通。20 Hzと60 Hzは定量採用外。",""]
    for folder,group in groupby(sorted(catalog,key=lambda item:item[1]),key=lambda item:item[0].folder):
        lines += ["## "+folder.as_posix(),""] + [f"- [{s.path.stem}]({p})" for s,p in group] + [""]
    (DEST/"README.md").write_text("\n".join(lines),encoding="utf-8")
    print(f"{len(catalog)} figures checked and saved.")


if __name__ == "__main__":
    main()

"""課題2の (2.5, 6.5) cm で測定した上り掃引の全点を描く。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "07_補足解析/課題2_位置依存/実験II/20261028_KIT_x2.5_z6.5_2.dat"
    target = root / "03_図表/測定点のみ" / (source.stem + ".png")
    frequency_hz, signal = np.loadtxt(source).T

    plt.rcParams["font.family"] = ["Yu Gothic", "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(8.4, 4.6), layout="constrained")
    ax.scatter(frequency_hz / 1e6, signal, s=15, color="#195880", linewidths=0)
    ax.set_title("課題2　測定波形（x = 2.5 cm, z = 6.5 cm）", loc="left", pad=14)
    ax.set_xlabel("周波数 (MHz)")
    ax.set_ylabel("ロックイン出力（記録単位）")
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_axisbelow(True)
    ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.02, 0.96, f"2026年10月28日　上り掃引・再測定2　{len(signal)}点",
            transform=ax.transAxes, va="top", fontsize=9)
    ax.margins(y=0.17)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    main()

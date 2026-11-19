"""課題1の周波数の保存刻みを調べる。2026-11-20 の測定記録に基づく再現用コード。"""
from pathlib import Path
import sys

import numpy as np

TASK1 = Path(__file__).resolve().parents[1] / "06_入力" / "課題1_共鳴信号" / "解析I"
sys.path.insert(0, str(TASK1))
from nmr_physical_models import fit_model


def main():
    frequency, signal = np.loadtxt(TASK1 / "20261020_KITtest01.dat").T
    reference = (frequency[signal.argmax()] + frequency[signal.argmin()]) / 2

    def center_hz(f):
        shape, _, _ = fit_model("voigt", (f - reference) / 1000, signal)
        return reference + 1000 * shape[0]

    center = center_hz(frequency)
    rng = np.random.default_rng(20261120)
    shifts = [center_hz(frequency + rng.uniform(-5, 5, frequency.size)) - center
              for _ in range(300)]
    print(f"300回の中心変化の標本標準偏差 = {np.std(shifts, ddof=1):.3f} Hz")
    print(f"全点を一律 +5 Hz としたときの中心変化 = {center_hz(frequency + 5) - center:+.2f} Hz")


if __name__ == "__main__":
    main()

"""Recreate the published figures/tables under 作業用; optionally refit all scans."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

from compare_asymmetry import PACKAGE, OUT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refit", action="store_true")
    args = parser.parse_args()
    code = Path(__file__).resolve().parent
    tables = PACKAGE / "04_解析結果"
    work = OUT.parent / "図表"
    work.mkdir(parents=True, exist_ok=True)
    if args.refit:
        for script in ("compare_symmetric.py", "voigt_uncertainties.py"):
            subprocess.run([sys.executable, "-B", str(code / script)], check=True)
    else:
        shutil.copyfile(tables / "4モデルの比較_全420件.csv", OUT / "symmetric_models.csv")
        name = "Voigt推定値と不確かさ.csv"
        shutil.copyfile(tables / name, work / name)
    subprocess.run([sys.executable, "-B", str(code / "voigt_figure_pack.py")], check=True)


if __name__ == "__main__":
    main()

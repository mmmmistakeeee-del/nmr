# レポートのコードと対応する図表

以下のコマンドは `NMR!` 直下で実行する。Python、NumPy、SciPy、pandas、Matplotlib を使用する。

[adopted_diagnostics.py](adopted_diagnostics.py) は保存した採用係数から直線背景込みの波形を再現し、課題2・3の全点残差から注意を判定する。隣接相関は測定順で1点ずらした残差のPearson相関で、絶対値が0.30を超えると「残差相関」。片側ピークのS/Nは背景除去後の山谷差の半分を残差標準偏差（残差平方和を自由度で割った平方根）で割り、5未満を「弱信号」とする。自由度は課題2で点数−6、課題3で点数−7。近い掃引端までの距離が採用モデルの吸収線半値全幅未満の場合は「背景範囲不足」。図の外輪・三角には、線幅の相対標準不確かさ25%以上と形状量の下限近傍も含む。120・180・240 Hzの背景構造は別欄の測定上の注記。参考・保留・除外点は選定時の記録に従う。課題1の個別図には全点と共鳴付近（±5 kHz）の隣接相関を別々に記載した。

| 対象 | コード・入力 | 実行方法・出力 |
|---|---|---|
| 全106波形の採用フィット | [plot_adopted_fits.py](plot_adopted_fits.py)、課題1の採用解JSON、課題2・3の4モデル比較CSVと主解析表 | `python -B 05_解析コード/plot_adopted_fits.py`。元波形を読み、採用係数で曲線と残差を再現する。出力は `03_図表/個別フィット`。 |
| 課題1の波形・不確かさ | [解析I](../06_入力/課題1_共鳴信号/解析I/) の `nmr_joint_noise.py`、`nmr_physical_models.py`、`nmr_uncertainty.py` | 採用解・300回の再推定の出力も同じフォルダ。課題1の元の図は [plot_report_figures.py](../06_入力/レポート/plot_report_figures.py) の `task1()`。 |
| 課題1の追加モデル比較 | [compare_symmetric.py](compare_symmetric.py) の `analyze` | 下記のコマンドで4モデルの全点・未使用区間のRMSを再計算できる。 |
| 課題2・3のモデルと比較 | [compare_asymmetry.py](compare_asymmetry.py) の `harmonic`、[survey.py](survey.py) の入力選択・最小二乗フィット、[compare_symmetric.py](compare_symmetric.py) の4モデル比較 | 全点残差RMSと2 kHzごとの区間予測RMSは `04_解析結果/4モデルの比較_全420件.csv`。 |
| 課題2・3の不確かさ、線幅、山谷間隔 | [voigt_uncertainties.py](voigt_uncertainties.py) | 各波形120回の再推定値を集計。半値全幅は変調前の吸収線から計算する。 |
| 本文の図1〜6に対応する波形・分布・条件別集計（本文順は fig1、fig2、fig3、fig4、fig6、fig5） | [voigt_figure_pack.py](voigt_figure_pack.py)、[reproduce.py](reproduce.py) | `python -B 05_解析コード/reproduce.py` で保存値から図表を再生成する。課題1の図は保存済み図を使用する。課題2・3の再フィットを含めるには `--refit` を付ける。出力は `作業用`。 |
| 基準解析・採否・測定日比較 | [課題2の解析II](../06_入力/課題2_位置依存/解析II/) の `fit_nmr_batch.py`、`position_plot_selection.py`、`compare_measurement_dates.py`。[課題3の解析III](../06_入力/課題3_変調条件と追加測定/解析III/) の `fit_conditions.py`、`compare_models.py`、`measurement_settings.json` | 基準解析を全試行からやり直す場合は、[補足解析](../07_補足解析/)に保存した各課題フォルダで実行する。個別図とレポート図表の再生成には同梱の採用データと保存済みCSVを使う。 |

課題1の追加比較は次のコマンドで表示できる。

```powershell
python -B -c "import sys; sys.path.insert(0,'05_解析コード'); from survey import inputs; from compare_symmetric import analyze; import pandas as pd; print(pd.DataFrame(analyze(next(inputs())))[['model','f0_Hz','full_rms','held_rms']])"
```

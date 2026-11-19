"""Rebuild the four presentation figures from the saved Task 2 fits."""

import csv
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np

from position_plot_selection import position_fit_display


HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results_with_additions'
METRICS = (
    ('f0', '共鳴周波数  f₀', 'MHz', 'viridis', (16, 16.03)),
    ('width', '吸収線の半値全幅  FWHM', 'kHz', 'cividis', (0, 16)),
    ('strength', '微分信号の山谷差', '記録値', 'viridis', (0, 3.5)),
)
STATUSES = (('ok', 'o', '警告なし'), ('warning', '^', '警告あり'))
INK = '#173752'


def load_points():
    with (RESULTS / 'fit_results.csv').open(encoding='utf-8-sig', newline='') as handle:
        rows = [(row, position_fit_display(row)) for row in csv.DictReader(handle)]
    return [dict(x=float(row['x']), z=float(row['z']), status=row['fit_status'], display=display,
                 f0=float(row['f0_Hz']) / 1e6, width=float(row['fwhm_Hz']) / 1000,
                 strength=9 * abs(float(row['L0'])) / (4 * np.sqrt(3) * float(row['hwhm_Hz'])))
            for row, display in rows if display != 'excluded']


def draw(points, split, three_d, output):
    groups = [('上側  z ≥ 4.5 cm', [p for p in points if p['z'] >= 4.5]),
              ('下側  z < 4.5 cm', [p for p in points if p['z'] < 4.5])] if split else [('', points)]
    view = '3D散布図' if three_d else 'xz分布図'
    scope = '上下別の' if split else '全測定域の'
    height = (10.2 if three_d else 8) if split else 5.6
    fig, axes = plt.subplots(len(groups), 3, figsize=(15.8, height), squeeze=False,
                             subplot_kw={'projection': '3d'} if three_d else {})
    fig.subplots_adjust(left=.064, right=.96, top=.83 if split else .81, bottom=.19,
                        wspace=.36, hspace=.74 if split else .25)
    fig.text(.05, .965, f'実験2　{scope}{view}', va='top', fontsize=20, weight='bold', color=INK)
    handles = [Line2D([], [], marker=marker, color=INK, ls='', label=label)
               for _, marker, label in STATUSES]
    handles.append(Line2D([], [], marker='o', markerfacecolor='none', color=INK, ls='', label='周波数の参考値'))
    if not three_d:
        handles.append(Line2D([], [], marker='x', color='#666666', ls='', label='推定保留'))
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(.97, .97), frameon=False, ncol=len(handles))
    for (group_label, part), row_axes in zip(groups, axes):
        if split:
            fig.text(.05, row_axes[0].get_position().y1 + .07, group_label,
                     weight='bold', color='#245a84')
        for ax, (key, title, unit, cmap, bounds) in zip(row_axes, METRICS):
            norm = plt.Normalize(*bounds)
            for status, marker, _ in STATUSES:
                group = [p for p in part if p['status'] == status and p['display'] == 'estimate']
                x, z, value = ([p[k] for p in group] for k in ('x', 'z', key))
                args = (x, z, value) if three_d else (x, z)
                ax.scatter(*args, c=value, cmap=cmap, norm=norm, marker=marker,
                           s=50 if three_d else 130, edgecolors='#344853', linewidths=.55)
            if key == 'f0':
                for point in (p for p in part if p['display'] == 'reference'):
                    coord = (point['x'], point['z'])
                    args = (*coord, point['f0']) if three_d else coord
                    ax.scatter(*args, marker='o', facecolors='none', edgecolors=INK,
                               s=65 if three_d else 130, linewidths=1.5)
            if not three_d:
                withheld = [p for p in part if p['display'] == 'withheld']
                ax.scatter([p['x'] for p in withheld], [p['z'] for p in withheld],
                           marker='x', c='#666666', s=100, linewidths=1.6)
            ax.set(xlabel='x (cm)', ylabel='z (cm)', title=f'{title} [{unit}]' if three_d else title)
            ax.set_xticks(np.arange(2.5, 7, 1))
            ax.set_xlim(2.25, 6.75)
            if three_d:
                ax.set_yticks(np.arange(2.5, 7, 1))
                ax.set_ylim(2.25, 6.75)
                ax.set_zlim(*bounds)
                ax.ticklabel_format(axis='z', useOffset=False)
                ax.view_init(elev=24, azim=-57)
            else:
                low, high = min(p['z'] for p in part), max(p['z'] for p in part)
                ax.set_yticks(np.arange(low, high + .1, .5))
                ax.set_ylim(low - .25, high + .25)
                ax.set_aspect('equal')
                ax.set_facecolor('#f4f8fb')
                ax.set_axisbelow(True)
                ax.grid(alpha=.22)
                bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                                   shrink=1 if not split else .88, pad=.04, label=unit)
                bar.ax.ticklabel_format(useOffset=False)
    note = '共鳴周波数には測定日間のずれが含まれる。'
    if three_d:
        note += ' 推定保留の位置は2D図に表示。'
    fig.text(.05, .04, note, color=INK, fontsize=10)
    fig.savefig(output, dpi=210)
    plt.close(fig)


def main():
    font = next(name for name in ('Yu Gothic', 'Meiryo', 'Noto Sans CJK JP')
                if name in font_manager.get_font_names())
    plt.rcParams.update({'font.family': font, 'font.size': 11})
    points = load_points()
    output = HERE / 'True_results'
    output.mkdir(exist_ok=True)
    files = []
    for view, three_d in (('2d_large_markers', False), ('3d', True)):
        for scope, split in (('all', False), ('split', True)):
            file = output / f'experiment2_{view}_{scope}.png'
            draw(points, split, three_d, file)
            files.append(file)
    with NamedTemporaryFile(dir=output, suffix='.zip', delete=False) as temporary:
        zip_path = Path(temporary.name)
    try:
        with ZipFile(zip_path, 'w', ZIP_DEFLATED) as archive:
            for file in files:
                archive.write(file, file.name)
        zip_path.replace(output / 'experiment2_graphs.zip')
    finally:
        zip_path.unlink(missing_ok=True)
    print('Task 2: updated four presentation figures with the saved display decisions.')


if __name__ == '__main__':
    main()

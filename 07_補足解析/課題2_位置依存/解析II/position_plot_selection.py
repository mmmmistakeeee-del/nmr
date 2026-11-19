"""Display decisions for individual saved sweeps in Task 2 position plots."""


SWEEP_DISPLAY = {
    '20261021_KIT_x3.0_z4.5.dat': 'excluded',
    '20261028_KIT_x2.5_z6.5_2.dat': 'withheld',
    '20261021_KIT_x2.5_z4.5_3.dat': 'reference',
}


def position_fit_display(row):
    """Reference fits appear only in maps of the field or resonance frequency."""
    if row['fit_status'] not in ('ok', 'warning'):
        return 'excluded'
    if float(row['x']) == 7 or float(row['z']) in (2, 7):
        return 'excluded'
    if row.get('segment') in (1, '1') and row.get('sweep') == 'up':
        return SWEEP_DISPLAY.get(row.get('filename'), 'estimate')
    return 'estimate'

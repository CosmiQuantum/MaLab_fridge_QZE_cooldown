"""Recover current/gain-matched linewidths from saved PUCQ4 spectra only."""
import argparse
import json
from pathlib import Path
import re
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_qspec_iq_series import decode_numeric, decode_text, joint_fit
from audit_pucq4_flux_qspec import newest_h5, selected_qspec_log

ROOT = Path(r'M:\_Data\20250822 - Olivia\pucq4_run_started_Oct_7\PUCQ4\pucq4_first_light')


def inspect(path, group_name, seed, current, origin, half_window=.8):
    with h5py.File(path, 'r') as handle:
        group = handle[group_name]
        x, i, q = [decode_numeric(group[name]) for name in ('Frequencies', 'I', 'Q')]
        assert len(x) == len(i) == len(q) and np.all(np.diff(x) > 0)
        assert all(np.isfinite(a).all() for a in (x, i, q))
        config = decode_text(group['Syst Config'])
        def number(key):
            values = re.findall(r"['\"]" + key + r"['\"]:\s*([-+0-9.eE]+)", config)
            return float(values[-1]) if values else None
        gain, length = number('qubit_gain_ge'), number('qubit_length_ge')
        assert gain is not None and length is not None
        fit = joint_fit(x, i, q, seed, half_window)
        step = float(np.median(np.diff(x)))
        samples = fit['fwhm_MHz'] / step
        resolved = bool(fit['vector_snr'] >= 4 and samples >= 2
                        and fit['fwhm_MHz'] < half_window * 1.5
                        and abs(fit['center_MHz'] - seed) < half_window * .7)
        quality = ('weak / tentative' if not resolved else
                   'coarse sampling' if samples < 4 else 'resolved')
        return dict(current_mA=current, origin=origin, h5=str(path),
                    qspec_gain=gain, qspec_length_us=length,
                    seed_frequency_MHz=float(seed),
                    center_MHz=fit['center_MHz'], fwhm_MHz=fit['fwhm_MHz'],
                    joint_IQ_SNR=fit['vector_snr'], residual_rms_sigma=fit['residual_rms_sigma'],
                    frequency_step_MHz=step, samples_per_FWHM=samples,
                    frequency_points=int(len(x)), frequency_start_MHz=float(x[0]),
                    frequency_stop_MHz=float(x[-1]),
                    quality=quality, resolved=resolved, fit_half_window_MHz=half_window,
                    readout_frequency_MHz=number('res_freq_ge'),
                    readout_gain=number('res_gain_ge'), readout_length_us=number('res_length'),
                    reps=number('reps'), rounds=number('rounds'),
                    acquisition_date=path.parents[3].name[:10],
                    method='Joint raw I/Q Lorentzian with separate linear backgrounds; no display smoothing')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    flux = ROOT / 'yoko_flux_campaign'
    narrow = json.loads((flux / 'flux_campaign_manifest.json').read_text())
    wide = json.loads((flux / 'single_sweep_qspec_manifest.json').read_text())
    report = dict(method='Same existing joint-IQ fitter for each actual acquired current/gain pair',
                  narrow_manifest=str(flux / 'flux_campaign_manifest.json'),
                  wide_manifest=str(flux / 'single_sweep_qspec_manifest.json'), devices={})
    for device, group in [('q4', 'Q5'), ('q6', 'Q6')]:
        rows = {}
        for key in sorted(wide['devices'][device]['map']['points'], key=float):
            current = float(key)
            w = wide['devices'][device]['map']['points'][key]
            observed_seed = float(w['peak_metrics']['frequency_MHz'])
            record = dict(wide=inspect(Path(w['h5']), group, observed_seed, current, 'wide map', 2.0))
            assert abs(record['wide']['qspec_gain'] - float(w['qspec_gain'])) < 1e-9
            n = narrow['devices'][device]['points'][key]
            log = selected_qspec_log(n)
            try:
                record['narrow'] = inspect(newest_h5(ROOT, log), group,
                                           float(n['qfreq_MHz']), current, 'selected local qspec')
            except (FileNotFoundError, KeyError, TypeError) as error:
                record['narrow'] = dict(available=False, reason=str(error))
            rows[key] = record
            print('%s %s | narrow %s | wide gain %.5g FWHM %.5f MHz %s' % (
                device, key,
                ('gain %.5g FWHM %.5f MHz %s' % (record['narrow']['qspec_gain'],
                  record['narrow']['fwhm_MHz'], record['narrow']['quality']))
                if 'fwhm_MHz' in record['narrow'] else 'not available',
                record['wide']['qspec_gain'], record['wide']['fwhm_MHz'], record['wide']['quality']), flush=True)
        report['devices'][device] = rows
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print('Saved ' + str(args.output), flush=True)


if __name__ == '__main__':
    main()

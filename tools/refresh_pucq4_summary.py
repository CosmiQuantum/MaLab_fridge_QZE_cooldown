"""Refresh selected zero-current rows and current/gain linewidth tables offline."""
import argparse
import copy
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

import h5py
import numpy as np
from pptx import Presentation
from pptx.opc.packuri import PackURI
from pptx.util import Inches, Pt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_pucq4_chip_summary as B
from audit_pucq4_m4 import audit
from analyze_qspec_iq_series import decode_numeric, joint_fit


def records_from_run(run, index):
    report = audit(run, index, 'zero_guard.json')
    entries = report['experiments']
    assert set(B.EXPERIMENTS) | {'ss_ge'} <= set(entries)
    for kind in ('qspec_ge', 'rabi_ge', 't1_ge', 't2_ge', 't2e_ge'):
        cal = entries[kind]['state_calibration']
        assert cal['matches_requested_shots'] and cal['population_reconstruction_verified']
    for kind in ('t1_ge', 't2_ge', 't2e_ge'):
        assert entries[kind]['independent']['accepted'] and entries[kind]['R2'] > .85
    assert entries['qspec_ge']['R2'] > .85 and entries['rabi_ge']['R2'] > .95
    assert .6 <= entries['t1_ge']['config']['pi_amp'] <= .7
    records = {}
    for kind, (_, folder) in B.EXPERIMENTS.items():
        entry = entries[kind]
        plots = list((run / 'documentation' / folder).glob('*Q_%d_*.png' % (index + 1)))
        assert plots
        record = dict(h5=entry['h5'], config=entry['config'],
                      plot=str(max(plots, key=lambda p: p.stat().st_mtime)),
                      values={}, date=run.name[:10], run=str(run))
        if kind == 'res_ge':
            record['values']['Found Freqs'] = [entry['found_frequency_MHz']]
        elif kind in ('t1_ge', 't2_ge', 't2e_ge'):
            name = {'t1_ge': 'T1', 't2_ge': 'T2', 't2e_ge': 'T2E'}[kind]
            record['values'] = {name: [entry['tau_us']], 'Errors': [entry['error_us']]}
        records[kind] = record
    records['ss_ge'] = dict(h5=entries['ss_ge']['h5'], config=entries['ss_ge']['config'],
                            values={'Fidelity': [entries['ss_ge']['contrast']]})
    ss_plots = list((run / 'documentation' / 'ss_repeat_meas_ge').glob(
        '**/*Q_%d_*Readout_Optimization*.png' % (index + 1)))
    assert len(ss_plots) == 1, ('Expected one SSF plot', run, ss_plots)
    records['ss_ge'].update(plot=str(ss_plots[0]), date=run.name[:10], run=str(run))
    return records, report


def linewidth_text(record):
    if record is None:
        return 'not measured'
    if not record.get('resolved', True):
        return 'unresolved*'
    suffix = '~' if record.get('quality') == 'coarse sampling' else ''
    return '%.3f%s' % (record['fwhm_MHz'], suffix)


def matching_linewidth(linewidths, device, current, gain, length, reports):
    row = '5' if device == 'q4' else '6'
    if current == 0 and row in reports:
        return reports[row]['experiments']['qspec_ge']
    pair = linewidths['devices'][device]['%+.1f' % current]
    matches = [pair[k] for k in ('narrow', 'wide')
               if 'fwhm_MHz' in pair[k]
               and np.isclose(pair[k]['qspec_gain'], gain, rtol=1e-6)
               and np.isclose(pair[k]['qspec_length_us'], length, rtol=1e-6)]
    matches.sort(key=lambda r: (not r['resolved'], r['quality'] == 'coarse sampling', r['origin'] == 'wide map'))
    return matches[0] if matches else None


def linewidth_averages(record):
    if record is None:
        return 'not recorded'
    if 'config' in record:
        p, count = record['config'], record['actual_axis']['points']
    else:
        p, count = record, record['frequency_points']
    return '%d x %d / %d' % (p['reps'], p['rounds'], count)


def replace_slides(prs, start, count, create):
    ids = prs.slides._sldIdLst
    old = list(ids)[start:start + count]
    before = len(prs.slides)
    create()
    assert len(prs.slides) == before + count
    new = list(ids)[-count:]
    for item in old:
        prs.part.drop_rel(item.rId)
        ids.remove(item)
    for item in new:
        ids.remove(item)
    for offset, item in enumerate(new):
        ids.insert(start + offset, item)
    # add_slide allocates from slide count. Renumber after removals so the next
    # replacement cannot reuse a still-live slide part's package name.
    for index, slide in enumerate(prs.slides, 1):
        slide.part.partname = PackURI('/ppt/slides/slide%d.xml' % index)


def replace_table(slide, headers, rows, font=10):
    old = next(shape for shape in slide.shapes if shape.has_table)
    y, height = old.top / 914400., old.height / 914400.
    old._element.getparent().remove(old._element)
    return B.table(slide, headers, rows, y=y, height=height, font_size=font)


def summary_slide(prs, records, widths):
    slide = B.slide_base(prs, 'PUCQ4 | six-qubit results at zero Yoko current',
                         'Python indices are zero-based. Frequencies/linewidths in MHz; coherence/errors in us. Dates differ.', records)
    rows = B.summary_rows(records)
    for row, values in zip(map(str, range(1, 7)), rows):
        values.insert(4, linewidth_text(widths[row]))
    headers = list(B.SUMMARY_HEADERS)
    headers.insert(4, 'Qspec FWHM MHz')
    B.table(slide, headers, rows, height=4.4, font_size=10)
    dates = '; '.join('M%s: %s' % (row, records[row]['t1_ge']['date']) for row in records)
    B.textbox(slide, dates + '\nM4/display Q4 has no assigned physical number. Physical Q4 is M5/Yoko 3; Q6 is M6/Yoko 4.\n'
              'Qspec gain and acquisition date are paired with each linewidth in the configuration/provenance tables.',
              .35, 5.65, 12.6, 1.0, 11)


def zero_settings(prs, records, reports, widths):
    B.zero_config_slides(prs, records, reports['4'])
    slide = prs.slides[-2]
    tab = next(s.table for s in slide.shapes if s.has_table)
    rows = [[cell.text for cell in row.cells] for row in tab.rows]
    rows[0].insert(4, 'FWHM MHz')
    for index, row in enumerate(rows[1:], 1):
        row.insert(4, linewidth_text(widths[str(index)]))
    replace_table(slide, rows[0], rows[1:], font=9)


def readout_slide(prs, records, reports):
    B.readout_table(prs, records, reports['4'])
    tab = next(s.table for s in prs.slides[-1].shapes if s.has_table)
    for row, report in reports.items():
        p = report['experiments']['t1_ge']['config']
        values = ['M' + row, '%.3f' % (p['res_freq_ge'] - p['res_freq_offset_ge']),
                  '%.3f' % p['res_freq_ge'], '%+.3f' % p['res_freq_offset_ge'],
                  '%.3f' % p['res_gain_ge'], '%.2f' % p['res_length'],
                  '%.2f' % (100 * report['experiments']['ss_ge']['contrast'])]
        for cell, value in zip(tab.rows[int(row)].cells, values):
            cell.text = value
            for para in cell.text_frame.paragraphs:
                para.font.name, para.font.size = 'Arial', Pt(12)


def refresh_yoko_tables(prs, linewidths, reports):
    provenance = {}
    for slide in prs.slides:
        titles = [s.text for s in slide.shapes if s.has_text_frame and s.top < Inches(.8)]
        if not titles or not titles[0].startswith('Yoko '):
            continue
        title = titles[0]
        device = 'q4' if title.startswith('Yoko 3') else 'q6'
        row = '5' if device == 'q4' else '6'
        tables = [s.table for s in slide.shapes if s.has_table]
        if not tables:
            if row in reports:
                foot = next(s for s in slide.shapes if s.has_text_frame and s.top > Inches(7))
                foot.text_frame.paragraphs[0].runs[0].text += ' Map retains its original campaign; the new 0 mA baseline is in the updated result/configuration tables.'
                foot.top, foot.height = Inches(6.98), Inches(.45)
                for p in foot.text_frame.paragraphs:
                    p.font.size = Pt(7)
            continue
        cells = [[c.text for c in r.cells] for r in tables[0].rows]
        headers, rows = cells[0], cells[1:]
        notes = {}
        for values in rows:
            current = float(values[0].split()[0])
            fresh = reports.get(row) if current == 0 else None
            if 'selected pulse settings' in title:
                if fresh:
                    p = fresh['experiments']['t1_ge']['config']
                    q = fresh['experiments']['qspec_ge']['config']
                    r = fresh['experiments']['rabi_ge']
                    values[1:] = ['%.6f' % p['qubit_freq_ge'], '%.3f' % (1000*p['sigma']),
                                  '%.6f' % p['pi_amp'], '%.5g' % q['qubit_gain_ge'],
                                  '%.3g' % q['qubit_length_ge'], '%d x %d / %d' %
                                  (r['config']['reps'], r['config']['rounds'], r['actual_axis']['points'])]
                match = matching_linewidth(linewidths, device, current, float(values[4]), float(values[5]), reports)
                notes[values[0]] = match
                values.insert(5, linewidth_text(match))
                values.insert(7, linewidth_averages(match))
            elif 'acquisition settings for the wide qspec map' in title:
                match = linewidths['devices'][device]['%+.1f' % current]['wide']
                assert np.isclose(float(values[1]), match['qspec_gain'], rtol=.001)
                values.insert(2, linewidth_text(match))
                notes[values[0]] = match
            elif fresh and 'optimized readout settings' in title:
                p = fresh['experiments']['t1_ge']['config']
                values[1:] = ['%.3f' % (p['res_freq_ge'] - p['res_freq_offset_ge']),
                              '%.3f' % p['res_freq_ge'], '%.3f' % p['res_freq_offset_ge'],
                              '%.3f' % p['res_gain_ge'], '%.2f' % p['res_length'],
                              '%.2f' % (100*fresh['experiments']['ss_ge']['contrast'])]
            elif fresh and 'measured coherence and averaging' in title:
                data = [fresh['experiments'][k] for k in ('t1_ge', 't2_ge', 't2e_ge')]
                values[1:] = ['%.3f +/- %.3f' % (d['tau_us'], d['error_us']) for d in data] + [
                    '%d x %d' % (d['config']['reps'], d['config']['rounds']) for d in data]
            elif fresh and 'coherence sampling and state calibration' in title:
                data = [fresh['experiments'][k] for k in ('t1_ge', 't2_ge', 't2e_ge')]
                values[1:] = ['%.3f-%.3f / %.4f' % (d['actual_axis']['start'], d['actual_axis']['stop'],
                              d['actual_axis']['step']) for d in data] + [
                              ' / '.join(str(d['state_calibration']['shots_per_state']) for d in data)]
        if 'selected pulse settings' in title:
            headers[1] = 'Calibrated f01 MHz'
            headers.insert(5, 'FWHM MHz')
            headers.insert(7, 'Width scan averages / N')
        elif 'acquisition settings for the wide qspec map' in title:
            headers.insert(2, 'FWHM MHz')
        replace_table(slide, headers, rows, font=9 if len(headers) >= 8 else 10)
        foot = next(s for s in slide.shapes if s.has_text_frame and s.top > Inches(7))
        if notes:
            text = 'Linewidths use the actual current/gain/pulse-length pair. * Too weak for a validated width; diagnostic candidates remain in notes. ~ Fewer than four samples per FWHM.'
        else:
            text = 'Zero-current baseline refreshed where stated; nonzero rows retain their original campaign. Exact settings and acquisition provenance are in notes.'
        if 'coherence sampling and state calibration' in title:
            text = 'Actual delay bounds/steps in us. Original campaign: T2 phase 0.5 MHz; relaxation 250 us.'
            if row in reports:
                e = reports[row]['experiments']
                text += ' Refreshed 0 mA: T2 phases %g / %g MHz; relaxation %g us.' % (
                    e['t2_ge']['config']['ramsey_freq'], e['t2e_ge']['config']['ramsey_freq'],
                    e['t1_ge']['config']['relax_delay'])
        elif 'optimized readout settings' in title:
            text = 'RO tone = base + offset. SSF is histogram contrast, not assignment accuracy. Nonzero rows retain their original campaign.'
        if row in reports and 'wide qspec' not in title:
            text += ' 0 mA updated ' + Path(reports[row]['run']).name + '.'
        foot.text_frame.paragraphs[0].runs[0].text = text
        foot.top, foot.height = Inches(6.98), Inches(.45)
        slide.notes_slide.notes_text_frame.text = json.dumps(dict(
            prior_notes=slide.notes_slide.notes_text_frame.text,
            linewidth_provenance=notes, zero_current_refresh=reports.get(row)), indent=2)
        provenance[title] = notes
    return provenance


def comparison_slide(prs, comparisons, failed_attempts=None):
    slide = B.slide_base(prs, 'Zero-current linewidth versus spectroscopy drive',
                         'Each row is one saved acquisition. Width errors are statistical fit errors; low-R2 rows are diagnostic. The requested target is 0.10 MHz.',
                         dict(measurements=comparisons, failed_attempts=failed_attempts or []))
    rows = []
    for report in comparisons:
        spec = report['experiments']['qspec_ge']
        uncertainty = (' +/- %.3f' % spec['fwhm_error_MHz']) if 'fwhm_error_MHz' in spec else ''
        rows.append([report['resonator_row'], '%.5g' % spec['config']['qubit_gain_ge'],
                     '%.3g' % spec['config']['qubit_length_ge'], '%.6f' % spec['center_MHz'],
                     '%.3f%s' % (spec['fwhm_MHz'], uncertainty),
                     '%.3f' % spec['peak_population'], '%.3f' % spec['R2'],
                     Path(report['run']).parent.name])
    assert len(rows) <= 12
    B.table(slide, ['Row', 'Qspec gain', 'Pulse us', 'f01 MHz', 'FWHM MHz', 'Peak Pe', 'R2', 'Run'],
            rows, height=5.6, font_size=9)
    if failed_attempts:
        B.textbox(slide, 'Excluded attempt: line_6c stopped at a fit-initialization error and saved placeholder HDF5; it was repeated as line_6d.',
                  .35, 6.65, 12.6, .3, 9)


def append_current_linewidth_tables(history, summary):
    for source in summary.slides:
        titles = [s.text for s in source.shapes if s.has_text_frame and s.top < Inches(.8)]
        if not titles or not titles[0].startswith('Yoko '):
            continue
        if not any(text in titles[0] for text in ('selected pulse settings', 'acquisition settings for the wide qspec map')):
            continue
        tab = next(s.table for s in source.shapes if s.has_table)
        cells = [[c.text for c in row.cells] for row in tab.rows]
        footer = next(s.text for s in source.shapes if s.has_text_frame and s.top > Inches(6.9))
        slide = B.slide_base(history, titles[0], footer)
        B.table(slide, cells[0], cells[1:], font_size=9)
        slide.notes_slide.notes_text_frame.text = source.notes_slide.notes_text_frame.text


def save_deck(prs, path, stamp, preview_dir=None):
    for index, slide in enumerate(prs.slides, 1):
        slide.part.partname = PackURI('/ppt/slides/slide%d.xml' % index)
        if slide.has_notes_slide:
            slide.notes_slide.part.partname = PackURI('/ppt/notesSlides/notesSlide%d.xml' % index)
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview_path = preview_dir / path.name
        prs.save(preview_path)
        assert len(Presentation(preview_path).slides) == len(prs.slides)
        with zipfile.ZipFile(preview_path) as archive:
            assert len(archive.namelist()) == len(set(archive.namelist()))
        return dict(path=str(preview_path), slides=len(prs.slides), preview=True)
    archive = path.parent / '_PowerPoint_archive'
    archive.mkdir(exist_ok=True)
    backup = archive / (path.stem + '_before_linewidth_' + stamp + '.pptx')
    pending = archive / (path.stem + '_linewidth_pending_' + stamp + '.pptx')
    assert not backup.exists() and not pending.exists()
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    prs.save(pending)
    assert len(Presentation(pending).slides) == len(prs.slides)
    with zipfile.ZipFile(pending) as check:
        assert len(check.namelist()) == len(set(check.namelist()))
    assert before == hashlib.sha256(path.read_bytes()).hexdigest(), 'Deck changed during report update'
    shutil.copy2(path, backup)
    pending.replace(path)
    return dict(path=str(path), backup=str(backup), slides=len(prs.slides),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--preview-dir', type=Path,
                        help='Write review copies and JSON here without changing either active deck')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding='utf-8-sig'))
    deck = B.ROOT / 'PUCQ4_chip_summary_20260908.pptx'
    assets = B.ROOT / 'PUCQ4_chip_summary_20260908_artifacts'
    prior_path = assets / 'summary_provenance.json'
    prior = json.loads(prior_path.read_text(encoding='utf-8-sig'))
    records = copy.deepcopy(prior['records'])
    reports = {'4': prior['M4_audit']}
    for row, run in plan['final_runs'].items():
        records[row], reports[row] = records_from_run(Path(run), int(row) - 1)
    B.enrich_records(records)
    linewidths = json.loads(Path(plan['flux_linewidths']).read_text())
    widths = {}
    for row, data in records.items():
        if row in reports:
            widths[row] = reports[row]['experiments']['qspec_ge']
            continue
        with h5py.File(data['qspec_ge']['h5'], 'r') as handle:
            g = handle['Q' + row]
            f, i, q = [decode_numeric(g[key]) for key in ('Frequencies', 'I', 'Q')]
        result = joint_fit(f, i, q, data['rabi_ge']['config']['qubit_freq_ge'], .8)
        widths[row] = {key: value for key, value in result.items() if np.isscalar(value)}
    prs = Presentation(deck)
    assert len(prs.slides) == 39, 'This one-time refresh expects the original 39-slide summary; do not apply it twice.'
    if plan.get('update_characterization', True):
        assert len(Presentation(B.ROOT / 'PUCQ4_first_light_characterization.pptx').slides) == 93, \
            'This one-time refresh expects the original 93-slide characterization deck.'
    replace_slides(prs, 0, 1, lambda: summary_slide(prs, records, widths))
    replace_slides(prs, 1, 1, lambda: B.m4_result_slide(prs, reports['4'], .10))
    replace_slides(prs, 2, 2, lambda: zero_settings(prs, records, reports, widths))
    replace_slides(prs, 4, 1, lambda: readout_slide(prs, records, reports))
    replace_slides(prs, 5, 3, lambda: B.sampling_slides(prs, records))
    for index, kind in enumerate(B.EXPERIMENTS):
        replace_slides(prs, 8 + index, 1, lambda kind=kind: B.gallery(prs, kind, records))
    spec_footer = next(s for s in prs.slides[9].shapes if s.has_text_frame and s.top > Inches(7))
    spec_footer.text_frame.paragraphs[0].runs[0].text = (
        'Original measured traces. M4/M6 refreshed at zero current; M5 qspec predates its final Ramsey correction. Exact dates and raw sources are in notes.')
    yoko_provenance = refresh_yoko_tables(prs, linewidths, reports)
    for row in sorted(plan['final_runs']):
        if row != '4':
            B.m4_result_slide(prs, reports[row], .10)
    if plan['final_runs']:
        ss_records = {row: records[row]['ss_ge'] for row in sorted(plan['final_runs'])}
        slide = B.slide_base(prs, 'Refreshed single-shot readout at zero Yoko current',
                             'Fresh prepared-state histograms after frequency/Rabi calibration. SSF is histogram contrast, not assignment accuracy.', ss_records)
        height = 6.1 / len(ss_records)
        for index, (row, record) in enumerate(ss_records.items()):
            y = .82 + index * height
            B.textbox(slide, 'M%s / index %d | %s | SSF contrast %.2f%%' %
                      (row, int(row) - 1, record['date'], 100 * record['values']['Fidelity'][0]),
                      .35, y, 12.6, .25, 11, True)
            B.picture(slide, record['plot'], .35, y + .27, 12.6, height - .36)
    comparisons = [json.loads(Path(path).read_text(encoding='utf-8-sig')) for path in plan['comparison_reports']]
    comparison_slide(prs, comparisons, plan.get('failed_attempts'))
    for slide in prs.slides:
        for shape in slide.shapes:
            assert shape.left >= 0 and shape.top >= 0
            assert shape.left + shape.width <= prs.slide_width + 100
            assert shape.top + shape.height <= prs.slide_height + 100
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    prior_backup = B.ROOT / '_PowerPoint_archive' / ('summary_provenance_before_linewidth_' + stamp + '.json')
    if args.preview_dir is None:
        assert not prior_backup.exists()
        shutil.copy2(prior_path, prior_backup)
    else:
        assets = args.preview_dir / 'artifacts'
        assets.mkdir(parents=True, exist_ok=True)
    output = save_deck(prs, deck, stamp, args.preview_dir)
    if plan.get('update_characterization', True):
        history_path = B.ROOT / 'PUCQ4_first_light_characterization.pptx'
        history = Presentation(history_path)
        replace_slides(history, 0, 1, lambda: summary_slide(history, records, widths))
        comparison_slide(history, comparisons, plan.get('failed_attempts'))
        for row in sorted(plan['final_runs']):
            B.m4_result_slide(history, reports[row], .10)
            for kind in list(B.EXPERIMENTS) + ['ss_ge']:
                record = records[row][kind]
                B.plot_slide(history, 'M%s / index %d | refreshed %s at zero current' %
                             (row, int(row) - 1, B.EXPERIMENTS[kind][0] if kind != 'ss_ge' else 'SSF'), [record['plot']],
                             'Normal one-pass round-robin output; exact acquisition configuration and HDF5 in notes.', record,
                             crop=B.RESONATOR_CROP if kind == 'res_ge' else None)
        append_current_linewidth_tables(history, prs)
        output['characterization'] = save_deck(history, history_path, stamp, args.preview_dir)
    report = {**prior, 'created': datetime.datetime.now().astimezone().isoformat(),
              'records': records, 'M4_audit': reports['4'], 'zero_current_refresh': reports,
              'linewidths': linewidths, 'zero_current_linewidths': widths,
              'yoko_linewidth_table_provenance': yoko_provenance,
              'linewidth_comparisons': comparisons, 'summary_slides': len(prs.slides),
              'failed_attempts': plan.get('failed_attempts', []),
              'linewidth_update': output, 'prior_provenance': str(prior_backup)}
    if args.preview_dir is None:
        prior_path.write_text(json.dumps(report, indent=2))
    (assets / 'chip_trace_sources.json').write_text(json.dumps(records, indent=2))
    (assets / 'linewidth_refresh.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()

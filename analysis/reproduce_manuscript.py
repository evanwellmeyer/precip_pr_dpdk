"""Rebuild manuscript figures without training or changing the HadGEM dataset."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from analysis.paths import research
parser = argparse.ArgumentParser()
parser.add_argument('--publish', action='store_true', help='Copy staged figures to the LaTeX folders')
args = parser.parse_args()
os.chdir(root)
os.environ['PYTHONPATH'] = str(root)
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
os.environ.setdefault('MPLBACKEND', 'Agg')
output = root / 'analysis/direct_revision'
logs = output / 'reproduction_logs'
logs.mkdir(exist_ok=True)
for folder in ['figures', 'supp_figures']:
    (output / folder).mkdir(exist_ok=True)
scripts = ['correlations.py', 'gridpoint_regression.py', 'gaussian_weighting.py',
           'method_figure.py', 'capacity.py', 'calibration.py', 'maps.py',
           'dTdP.py', 'cv_figure.py', 'amip_target.py', 'amip_test.py',
           'archive_figure.py', 'climatology.py']
record = {'started': datetime.now(timezone.utc).isoformat(), 'scripts': {}}
inputs = [research / 'HadGEM' / name for name in
          ['GA789_PR_bilinear_rg128.nc', 'GA789_dPdK_bilinear_rg128.nc',
           'hadgem_landmask_rg128.nc']]
for folder in ['direct_dpdk_bilinear', 'direct_dpdk_bilinear_cv']:
    base = research / 'weights' / folder
    for pattern in ['**/*.pth', '**/norm_stats.json', '**/born_bins.json',
                    '**/data_splits.npz', '**/test_results.npz']:
        inputs.extend(sorted(base.glob(pattern)))
record['inputs'] = {}
for path in inputs:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b''):
            digest.update(block)
    record['inputs'][str(path)] = digest.hexdigest()
for script in scripts:
    path = output / script
    print(f'Running {script}', flush=True)
    with (logs / f'{path.stem}.log').open('w') as log:
        subprocess.run([sys.executable, '-u', str(path)], stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    record['scripts'][script] = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f'Finished {script}', flush=True)

record['figures'] = {}
for folder in ['figures', 'supp_figures']:
    for staged in sorted((output / folder).iterdir()):
        if staged.suffix.lower() not in {'.png', '.pdf'}:
            continue
        filename = staged.relative_to(output)
        record['figures'][str(filename)] = hashlib.sha256(staged.read_bytes()).hexdigest()
        if args.publish:
            paper = root / 'AMS LaTeX Package V6.1'
            (paper / filename).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(staged, paper / filename)
record['finished'] = datetime.now(timezone.utc).isoformat()
record['published'] = args.publish
(output / 'reproduction_manifest.json').write_text(json.dumps(record, indent=2) + '\n')
print('All manuscript figure scripts completed.', flush=True)

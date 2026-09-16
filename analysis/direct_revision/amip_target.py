from pathlib import Path
from analysis.paths import research, project
import json
import numpy as np
import xarray as xr
import xesmf as xe

root = research / 'AMIP'
output = Path('analysis/direct_revision')
labels = [
    'BCC-CSM2-MR_r1i1p1f1', 'CESM2_r1i1p1f1', 'CNRM-CM6-1_r1i1p1f2',
    'CanESM5_r1i1p2f1', 'E3SM-1-0_r2i1p1f1', 'GFDL-CM4_r1i1p1f1',
    'HadGEM3-GC31-LL_r5i1p1f3', 'IPSL-CM6A-LR_r1i1p1f1',
    'MIROC6_r1i1p1f1', 'MRI-ESM2-0_r1i1p1f1', 'TaiESM1_r1i1p1f1',
]
with xr.open_dataset(research / 'HadGEM/GA789_PR_bilinear_rg128.nc') as dataset:
    latitude = dataset.latitude.values
    longitude = dataset.longitude.values
destination = xr.Dataset(coords={'lat': latitude, 'lon': longitude})
responses = []
inputs = []
warming = []
records = []

for label in labels:
    model, member = label.rsplit('_', 1)
    paths = [
        root / 'pd_clim' / f'pr_{model}_amip_{member}_clim.nc',
        root / 'future4K/pr_clim' / f'pr_{model}_amip-future4K_{member}_clim.nc',
        root / 'PD_tas_clim' / f'tas_{model}_amip_{member}_clim.nc',
        root / 'future4K/tas_clim' / f'tas_{model}_amip-future4K_{member}_clim.nc',
    ]
    fields = []
    for number, path in enumerate(paths):
        with xr.open_dataset(path) as dataset:
            if '1985-2014' not in dataset.attrs.get('history', ''):
                raise ValueError(f'Unverified climatology period: {path}')
            if number in [1, 3] and dataset.attrs['experiment_id'] != 'amip-future4K':
                raise ValueError(f'Unexpected warming experiment: {path}')
            if dataset.attrs['source_id'] != model or dataset.attrs['variant_label'] != member:
                raise ValueError(f'Model or realization mismatch: {path}')
            field = dataset['pr' if number < 2 else 'tas'].squeeze(drop=True).load()
        field = field.assign_coords(lon=((field.lon + 180) % 360) - 180).sortby('lon')
        fields.append(field)
    present, future, presenttemperature, futuretemperature = fields
    present, future = xr.align(present, future, join='exact')
    presentglobal = presenttemperature.weighted(np.cos(np.deg2rad(presenttemperature.lat))).mean(('lat', 'lon'))
    futureglobal = futuretemperature.weighted(np.cos(np.deg2rad(futuretemperature.lat))).mean(('lat', 'lon'))
    delta = float(futureglobal - presentglobal)
    if delta <= 0:
        raise ValueError(f'Nonpositive warming for {label}')
    response = (future - present) * (365.25 * 86400) / delta
    weightpath = output / f'amip_bilinear_weights_{model}.nc'
    regridder = xe.Regridder(present, destination, 'bilinear', periodic=True,
                            extrap_method='nearest_s2d', filename=str(weightpath),
                            reuse_weights=weightpath.exists())
    response = regridder(response).values.astype(np.float32)
    precipitation = regridder(present * (365.25 * 86400)).values.astype(np.float32)
    if not np.isfinite(response).all() or not np.isfinite(precipitation).all():
        raise ValueError(f'Nonfinite input or response for {label}')
    responses.append(response)
    inputs.append(precipitation)
    warming.append(delta)
    records.append({'model': label, 'warming': delta, 'sources': [str(p) for p in paths]})
    print(label, delta, flush=True)

dataset = xr.Dataset(
    {'dPdK': (('realization', 'latitude', 'longitude'), np.stack(responses)),
     'PR': (('realization', 'latitude', 'longitude'), np.stack(inputs)),
     'temperature_change': ('realization', warming)},
    coords={'realization': labels, 'latitude': latitude, 'longitude': longitude},
    attrs={'period': '1985-2014', 'experiment': 'amip-future4K minus amip',
           'definition': 'precipitation climatology difference divided by normalized cosine weighted global temperature change',
           'regrid_method': 'bilinear with periodic longitude and nearest extrapolation'})
dataset.dPdK.attrs['units'] = 'mm yr-1 K-1'
dataset.PR.attrs['units'] = 'mm yr-1'
dataset.to_netcdf(output / 'amip_direct.nc')
with open(output / 'amip_sources.json', 'w') as file:
    json.dump(records, file, indent=2)

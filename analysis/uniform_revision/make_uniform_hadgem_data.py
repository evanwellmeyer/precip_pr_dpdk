import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import xesmf as xe
from analysis.paths import hadgem


parser = argparse.ArgumentParser()
parser.add_argument(
    "--source-dir",
    default=str(hadgem),
)
parser.add_argument(
    "--output-dir",
    default=str(hadgem),
)
parser.add_argument("--overwrite", action="store_true")
args = parser.parse_args()

sourcepath = Path(args.source_dir)
outputpath = Path(args.output_dir)
projectpath = Path(__file__).resolve().parents[2]
weightpath = projectpath / "analysis/uniform_revision/hadgem_bilinear_weights.nc"

prpath = outputpath / "GA789_PR_bilinear_rg128.nc"
dpdkpath = outputpath / "GA789_dPdK_bilinear_rg128.nc"
secondsperyear = 365.25 * 24 * 60 * 60

families = [
    {
        "name": "GA7",
        "prpresent": "GA7_pr_his_clim.nc",
        "prwarming": "GA7_pr_fut_clim.nc",
        "tpresent": "GA7_ts_his_clim.nc",
        "twarming": "GA7_ts_fut_clim.nc",
        "tvariable": "surface_temperature",
    },
    {
        "name": "GA8",
        "prpresent": "GA8_pr_his_clim.nc",
        "prwarming": "GA8_pr_fut_clim.nc",
        "tpresent": "GA8_tas_his_clim.nc",
        "twarming": "GA8_tas_fut_clim.nc",
        "tvariable": "air_temperature",
    },
    {
        "name": "GA9",
        "prpresent": "GA9_pr_his_clim.nc",
        "prwarming": "GA9_pr_fut_clim.nc",
        "tpresent": "GA9_tas_his_clim.nc",
        "twarming": "GA9_tas_fut_clim.nc",
        "tvariable": "air_temperature",
    },
]

if not args.overwrite:
    existing = [path for path in [prpath, dpdkpath] if path.exists()]
    if existing:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output already exists: {names}")

outputpath.mkdir(parents=True, exist_ok=True)
weightpath.parent.mkdir(parents=True, exist_ok=True)

targetlatitude = np.linspace(-90.0, 90.0, 128)
targetlongitude = np.linspace(-180.0, 180.0, 192, endpoint=False)
targetgrid = xr.Dataset(
    coords={"latitude": targetlatitude, "longitude": targetlongitude}
)

prfields = []
dpdkfields = []
temperaturechanges = []
labels = []
sourceids = []
familylabels = []
regridder = None
sourcegrid = None

for family in families:
    name = family["name"]
    print(f"Loading {name}", flush=True)

    with xr.open_dataset(sourcepath / family["prpresent"]) as dataset:
        prpresent = dataset["precipitation_flux"].load()
    with xr.open_dataset(sourcepath / family["prwarming"]) as dataset:
        prwarming = dataset["precipitation_flux"].load()
    with xr.open_dataset(sourcepath / family["tpresent"]) as dataset:
        tpresent = dataset[family["tvariable"]].load()
    with xr.open_dataset(sourcepath / family["twarming"]) as dataset:
        twarming = dataset[family["tvariable"]].load()

    fields = [prpresent, prwarming, tpresent, twarming]
    referenceids = prpresent.realization.values
    referencelatitude = prpresent.latitude.values
    referencelongitude = prpresent.longitude.values

    for field in fields[1:]:
        if not np.array_equal(field.realization.values, referenceids):
            raise ValueError(f"Realization identifiers differ within {name}")
        if not np.array_equal(field.latitude.values, referencelatitude):
            raise ValueError(f"Latitude coordinates differ within {name}")
        if not np.array_equal(field.longitude.values, referencelongitude):
            raise ValueError(f"Longitude coordinates differ within {name}")

    currentgrid = xr.Dataset(
        coords={
            "latitude": referencelatitude,
            "longitude": referencelongitude,
        }
    )
    if sourcegrid is None:
        sourcegrid = currentgrid
        regridder = xe.Regridder(
            sourcegrid,
            targetgrid,
            "bilinear",
            periodic=True,
            extrap_method="nearest_s2d",
            filename=str(weightpath),
            reuse_weights=weightpath.exists(),
        )
    else:
        if not np.array_equal(currentgrid.latitude, sourcegrid.latitude):
            raise ValueError(f"The {name} latitude grid differs from the other PPEs")
        if not np.array_equal(currentgrid.longitude, sourcegrid.longitude):
            raise ValueError(f"The {name} longitude grid differs from the other PPEs")

    latitudeweights = np.cos(np.deg2rad(tpresent.latitude))
    presentglobal = tpresent.weighted(latitudeweights).mean(
        ("latitude", "longitude")
    )
    warmingglobal = twarming.weighted(latitudeweights).mean(
        ("latitude", "longitude")
    )
    temperaturechange = warmingglobal - presentglobal

    if not np.all(np.isfinite(temperaturechange.values)):
        raise ValueError(f"Nonfinite temperature change in {name}")
    if float(temperaturechange.min()) <= 0.0:
        raise ValueError(f"Nonpositive temperature change in {name}")

    prphysical = prpresent * secondsperyear
    dpdknative = (prwarming - prpresent) * secondsperyear / temperaturechange

    prregridded = regridder(prphysical).transpose(
        "realization", "latitude", "longitude"
    )
    dpdkregridded = regridder(dpdknative).transpose(
        "realization", "latitude", "longitude"
    )

    newlabels = [f"{name}_{index + 1}" for index in range(len(referenceids))]
    prregridded = prregridded.assign_coords(realization=newlabels)
    dpdkregridded = dpdkregridded.assign_coords(realization=newlabels)

    prfields.append(prregridded.astype(np.float32).load())
    dpdkfields.append(dpdkregridded.astype(np.float32).load())
    temperaturechanges.extend(temperaturechange.values.astype(float).tolist())
    labels.extend(newlabels)
    sourceids.extend(referenceids.astype(str).tolist())
    familylabels.extend([name] * len(referenceids))

    print(
        f"{name}: {len(referenceids)} members, "
        f"median warming {float(temperaturechange.median()):.3f} K",
        flush=True,
    )

pr = xr.concat(prfields, dim="realization")
dpdk = xr.concat(dpdkfields, dim="realization")

pr.name = "PR"
pr.attrs = {
    "long_name": "present day annual mean precipitation climatology",
    "units": "mm yr-1",
}
dpdk.name = "dPdK"
dpdk.attrs = {
    "long_name": "precipitation change per degree of global mean warming",
    "units": "mm yr-1 K-1",
    "definition": "(warming precipitation - present precipitation) / global mean temperature change",
}

commoncoords = {
    "realization": labels,
    "source_realization": ("realization", sourceids),
    "ppe_family": ("realization", familylabels),
}
pr = pr.assign_coords(commoncoords)
dpdk = dpdk.assign_coords(commoncoords)

prdataset = pr.to_dataset()
prdataset.attrs = {
    "source": "HadGEM GA7, GA8, and GA9 paired five year climatologies",
    "construction": "Physical present day precipitation was converted from kg m-2 s-1 to mm yr-1 before regridding.",
    "regridding": "Periodic bilinear interpolation from the common N96 grid to 128 by 192, with nearest source extrapolation at unmapped polar points.",
}

dpdkdataset = dpdk.to_dataset()
dpdkdataset["temperature_change"] = xr.DataArray(
    np.asarray(temperaturechanges, dtype=np.float32),
    dims="realization",
    coords={"realization": labels},
    attrs={
        "long_name": "cosine latitude weighted global mean surface temperature change",
        "units": "K",
    },
)
dpdkdataset.attrs = {
    "source": "HadGEM GA7, GA8, and GA9 paired five year climatologies",
    "construction": "The physical precipitation difference was divided by each member's cosine latitude weighted global mean temperature change on the native grid before regridding.",
    "regridding": "Periodic bilinear interpolation from the common N96 grid to 128 by 192, with nearest source extrapolation at unmapped polar points.",
}

expectedshape = (1515, 128, 192)
if pr.shape != expectedshape or dpdk.shape != expectedshape:
    raise ValueError(f"Unexpected output shapes: PR {pr.shape}, dPdK {dpdk.shape}")
if not np.array_equal(pr.realization.values, dpdk.realization.values):
    raise ValueError("PR and dPdK realization labels differ")
if not np.all(np.isfinite(pr.values)) or not np.all(np.isfinite(dpdk.values)):
    raise ValueError("The output contains nonfinite values")
if float(pr.min()) < 0.0:
    raise ValueError("Bilinear regridding produced negative precipitation")

legacyinput = sourcepath / "GA789_PR_his_rg128.nc"
if legacyinput.exists():
    with xr.open_dataset(legacyinput) as dataset:
        legacylabels = dataset.realization.values.astype(str)
    if not np.array_equal(pr.realization.values.astype(str), legacylabels):
        raise ValueError("The new realization order differs from the existing split order")

prseam = float(abs(pr.isel(longitude=0) - pr.isel(longitude=-1)).mean())
printerior = float(abs(pr.diff("longitude")).mean())
dpdkseam = float(abs(dpdk.isel(longitude=0) - dpdk.isel(longitude=-1)).mean())
dpdkinterior = float(abs(dpdk.diff("longitude")).mean())

print(f"PR longitude seam ratio: {prseam / printerior:.3f}")
print(f"dPdK longitude seam ratio: {dpdkseam / dpdkinterior:.3f}")
print(f"PR range: {float(pr.min()):.3f} to {float(pr.max()):.3f} mm yr-1")
print(
    f"dPdK range: {float(dpdk.min()):.3f} to {float(dpdk.max()):.3f} "
    "mm yr-1 K-1"
)

for dataset, path, variables in [
    (prdataset, prpath, ["PR"]),
    (dpdkdataset, dpdkpath, ["dPdK", "temperature_change"]),
]:
    temporarypath = path.with_suffix(".tmp.nc")
    encoding = {
        variable: {"dtype": "float32", "zlib": True, "complevel": 4}
        for variable in variables
    }
    dataset.to_netcdf(temporarypath, encoding=encoding)
    temporarypath.replace(path)
    print(f"Saved {path}")

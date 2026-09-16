from pathlib import Path
from analysis.paths import research, project
import json
import numpy as np
import torch
import xarray as xr
from unet import ProbUNet

output = Path('analysis/direct_revision')
modelpath = Path(str(research / "weights/direct_dpdk_bilinear/unet_direct_dpdk_flat_ch64_k3_bins64_min-700_max1200_sigma0.6_dropout0.1"))
with xr.open_dataset(output / 'amip_direct.nc') as dataset:
    inputs = dataset.PR.values.astype(np.float32)[:, None]
    targets = dataset.dPdK.values.astype(np.float32)
    labels = dataset.realization.values.astype(str)
    latitude = dataset.latitude.values
with xr.open_dataset(str(research / "HadGEM/hadgem_landmask_rg128.nc")) as dataset:
    mask = dataset.land_mask.values.astype(bool)
with xr.open_dataset(str(research / "HadGEM/GA789_dPdK_bilinear_rg128.nc")) as dataset:
    hadgemtargets = dataset.dPdK.values.astype(np.float32)
splits = np.load(modelpath / 'data_splits.npz')
trainindices = splits['train']
if len(trainindices) != 1060:
    raise ValueError('The selected model has an unexpected training set size')
ppemean = hadgemtargets[trainindices].mean(axis=0)
with open(modelpath / 'norm_stats.json') as file:
    normalization = json.load(file)
with open(modelpath / 'born_bins.json') as file:
    bins = json.load(file)
inputs = (inputs - normalization['x_mean'][0]) / normalization['x_std'][0]
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
centers = torch.tensor(bins['bin_centers_norm'], device=device)[None, :, None, None]
predictions = []
model = ProbUNet(1, 64, 3, 0.1, 64, gn_groups=1).to(device)
model.eval()
for seed in range(10):
    checkpoint = torch.load(modelpath / f'{modelpath.name}_member{seed}.pth', map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    with torch.inference_mode():
        batches = []
        for start in range(0, len(inputs), 4):
            probabilities = model(torch.tensor(inputs[start:start+4], device=device))
            prediction = (probabilities * centers).sum(dim=1)
            prediction = prediction * normalization['y_std'] + normalization['y_mean']
            batches.append(prediction.cpu().numpy())
        predictions.append(np.concatenate(batches))
prediction = np.mean(predictions, axis=0)
amipmean = (targets.sum(axis=0)[None] - targets) / (len(targets) - 1)
weights = np.cos(np.deg2rad(latitude))[:, None] * np.ones(targets.shape[1:])
landweights = weights * mask
records = []
for number, label in enumerate(labels):
    record = {'model': label}
    for name, weight in [('global', weights), ('land', landweights)]:
        nnrmse = np.sqrt(np.sum((prediction[number]-targets[number])**2*weight)/weight.sum())
        amipmeanrmse = np.sqrt(np.sum((amipmean[number]-targets[number])**2*weight)/weight.sum())
        ppemeanrmse = np.sqrt(np.sum((ppemean-targets[number])**2*weight)/weight.sum())
        record[name] = {
            'nn_rmse': float(nnrmse),
            'amip_mean_rmse': float(amipmeanrmse),
            'ppe_mean_rmse': float(ppemeanrmse),
            'improvement_vs_amip_mean_percent': float(100*(1-nnrmse/amipmeanrmse)),
            'improvement_vs_ppe_mean_percent': float(100*(1-nnrmse/ppemeanrmse)),
        }
    records.append(record)
with open(output / 'amip_test.json', 'w') as file:
    json.dump(records, file, indent=2)
np.savez_compressed(
    output / 'amip_predictions.npz',
    prediction=prediction,
    amipmean=amipmean,
    ppemean=ppemean,
    truth=targets,
    labels=labels,
)
print(json.dumps(records, indent=2))

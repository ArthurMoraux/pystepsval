# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 17:46:19 2025

@author: u0168535
"""

import os
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

scores_path = 'C:/Users/u0168535/.conda/envs/pysteps_dev/pystepsval/refactoring_arthur/test_data/scores/'
scores = xr.open_dataset(f'{scores_path}/scores.nc')

reference_time = scores.forecast_reference_time.values[0]

save_dir_plot = f'{scores_path}/plots'
if not os.path.exists(save_dir_plot):
    os.makedirs(save_dir_plot)

deterministic_metrics = [
    'RMSE_deterministic',
    'ME_deterministic',
    'MAPE_deterministic',
    'FSS_probabilistic',
    'Brier',
    'CRPS',
    ]

ensemble_metrics = [
    'RMSE_probabilistic',
    'ME_probabilistic',
    'MAPE_probabilistic',
    'FSS_per_member',
    'POD',
    'FAR',
    'ETS',
    'CSI',
    'RankHist',
    'Reliability',
    'Histogram',
    ]

spatial_metrics = [
    'FSS_probabilistic',
    'FSS_per_member',
    'POD',
    'FAR',
    'ETS',
    'CSI',
    ]

zero_metrics = [
    'RMSE_deterministic',
    'ME_deterministic',
    'MAPE_deterministic',
    'RMSE_probabilistic',
    'ME_probabilistic',
    'MAPE_probabilistic',
    'Brier',
    'CRPS',
    'FAR',
    ]

one_metrics = [
    'FSS_probabilistic',
    'FSS_per_member',
    'POD',
    'ETS',
    'CSI',
    ]

timesteps = scores.sizes['time']
ens_size = scores.sizes['ens_number']
xx = [0.,0.1,0.2,0.3,0.4,0.55,0.7,0.8,0.9,1.]
plot_thresholds = [0.1, 0.5, 1.0, 5.0]
plot_leadtimes = np.array(timesteps) #[5,30,60,120,180]
plot_ileadtimes = np.array(plot_leadtimes)/5 - 1
plot_window_sizes = [1, 5, 11, 21]


plot_metrics = [
    'RMSE_deterministic',
    'ME_deterministic',
    'MAPE_deterministic',
    'RMSE_probabilistic',
    'ME_probabilistic',
    'MAPE_probabilistic',
    'FSS_probabilistic',
    'FSS_per_member',
    'Brier',
    'CRPS',
    'POD',
    'FAR',
    'ETS',
    'CSI',
    ]

for metric in plot_metrics:
    plot_window_sizes = [5, 10, 30, 60] if metric in spatial_metrics else [None]
    
    metric_data = scores[metric]
    fig, ax = plt.subplots(figsize=(8,6), dpi=300)
    for ith, th in enumerate(plot_thresholds):
        for iws, ws in enumerate(plot_window_sizes):
            # best_metric = np.full((1,timesteps), -np.inf) if metric in one_metrics else np.full((1,timesteps), np.inf)
            # best_metric_name = np.full((1,timesteps), '', dtype=object)
            print(metric, ith, iws)
            
            # best_nwc_method_name = np.full((1,timesteps), f'{nwc_name}{nwc_info}')
            # real_metric = metric_data[iws,ith].values if metric in spatial_metrics else metric_data[ith].values
            # comparison_metric = best_metric < real_metric if metric in one_metrics else best_metric > real_metric
            # best_metric_name[comparison_metric] = best_nwc_method_name[comparison_metric]
            # best_metric[comparison_metric] = real_metric[comparison_metric]
            
            data = metric_data.isel(window_size=iws,threshold=ith) if metric in spatial_metrics else metric_data.isel(threshold=ith)
            wsth = f'{ws},{th}' if metric in spatial_metrics else f'{th}'
            if metric in deterministic_metrics:
                data.plot(label=wsth)
            else:
                data.mean(dim='ens_number').plot(label=f'{wsth}')
                for member in range(ens_size):
                    if member == 0:
                        data.isel(ens_number=member).plot(alpha=0.05) #label=f'{wsth}')
                    else:
                        last_line = plt.gca().get_lines()[-1]
                        last_color = last_line.get_color()
                        data.isel(ens_number=member).plot(c=last_color, alpha=0.05)
            plt.xlabel('Leadtime [min]')
    plt.legend(title='ws,th' if metric in spatial_metrics else 'th', ncols=1, fontsize='x-small')
    plt.savefig(f'{save_dir_plot}/{metric}.png', bbox_inches='tight', dpi=300)

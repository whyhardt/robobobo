import os.path

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


if __name__ == "__main__":
    downsample_rate = 10
    plot = True
    plot_index = 500
    save = True
    dataset_path = r'..\stock_data\stocks_sp500_2010_2020_mvgavg50.csv'
    plot_window_len = 1

    # load dataset
    dataset = pd.read_csv(dataset_path, index_col=0)
    x = dataset.to_numpy()
    t = np.linspace(0, 1, x.shape[0])

    # down sampling by interpolation
    seq_len = x.shape[0]
    target_seq_len = x.shape[0] // downsample_rate
    t_target = np.linspace(0, 1, target_seq_len)
    x_interp = np.zeros([target_seq_len, x.shape[1]])
    for i in range(x.shape[1]):
        x_interp[:, i] = np.interp(t_target, t, x[:, i])

    # downsampling by taking every n-th sample
    x_downsampled = x[::downsample_rate, :]
    t_downsampled = t[::downsample_rate]
    # get date index from original dataset and take every n-th date
    index_downsampled = dataset.index[::downsample_rate]

    if plot:
        plt.plot(t, x[:, plot_index], label=f"Original length {seq_len}")
        plt.plot(t_target, x_interp[:, plot_index], label=f"Interpolated length {target_seq_len}")
        plt.plot(t_downsampled, x_downsampled[:, plot_index], label=f"Downsampled length {target_seq_len}")
        plt.legend()
        plt.xlabel('Time')
        plt.title(f"Interpolation with downsample rate {downsample_rate}")
        if plot_window_len > 0:
            plt.xlim([.8, 1.])
        plt.show()

    if save:
        dataset_name = os.path.basename(dataset_path).split('.')[0]
        dataset_path = os.path.dirname(dataset_path)
        # df = pd.DataFrame(x_interp, index=range(target_seq_len), columns=dataset.columns)
        # df.to_csv(os.path.join(dataset_path, f"{dataset_name}_downsampled{downsample_rate}_new.csv"))
        df = pd.DataFrame(x_downsampled, index=index_downsampled, columns=dataset.columns)
        df.to_csv(os.path.join(dataset_path, f"{dataset_name}_downsampled{downsample_rate}_new.csv"))
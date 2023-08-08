import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, Normalizer
from matplotlib import pyplot as plt


class MultivariateTimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, normalize=False, standardize=False, differentiate=False, start_zero=False, shuffle=False):
        self.seq_len = seq_len
        self.standardize = standardize
        self.differentiate = differentiate
        self.normalize = normalize
        self.start_zero = start_zero
        self.scaler = StandardScaler()
        self.min = None
        self.max = None
        self.data = self._process_data(data, shuffle=shuffle)

    def __getitem__(self, index):
        # start_index = index
        # end_index = index + self.seq_len
        # return self.data[start_index:end_index, :, :]
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]  # - self.seq_len + 1

    def _slice_sequence(self, sequence):
        slices = []
        num_slices = (len(sequence) - self.seq_len) + 1
        for i in range(num_slices):
            slices.append(sequence[i:i+self.seq_len, :])
        if len(slices) > 1:
            slices = torch.stack(slices[:-1], dim=0)
        else:
            slices = slices[0].unsqueeze(0)
        print(slices.shape)
        return slices

    def _process_data(self, data, shuffle=False):
        # convert the Date column to a datetime object and set it as the index
        # data['Date'] = pd.to_datetime(data['Date'])
        # data.set_index('Date', inplace=True)

        # differentiate the data if required
        if self.differentiate:
            data = data.diff().dropna()
            print("Data differentiated!")

        # make data to tensor
        data = torch.Tensor(data.to_numpy()).float()

        # slice the data into sequences
        if self.seq_len > 0:
            data = self._slice_sequence(data)
            print(f"Data sliced into sequences of length {self.seq_len}!")
        else:
            data = data.unsqueeze(0)

        # start from zero
        if self.start_zero:
            data -= data[:, 0, :].unsqueeze(1).repeat(1, data.shape[1], 1)
            print(f"Sequences start now from 0!")

        # standardize the data if required
        for i, d in enumerate(data):
            self.scaler = self.scaler.partial_fit(d)
            if self.standardize:
                data[i] = torch.Tensor(self.scaler.transform(d)).float()
        if self.standardize:
            print("Data standardized!")

        # normalize the data if required
        for i, d in enumerate(data):
            if self.normalize:
                batch_max = torch.abs(d).max(dim=0)[0]
                data[i, :, batch_max!=0] = d[:, batch_max != 0] / batch_max[batch_max != 0]

        # convert the data another time
        data = torch.Tensor(data).float()

        # shuffle the data along the first dimension
        if shuffle:
            data = data[torch.randperm(data.shape[0])]
            print("Data shuffled along the batch dimension!")

        return data


def create_dataloader(training_data, seq_len, batch_size, train_ratio,
                      normalize=False, standardize=False,
                      differentiate=False, start_zero=False, **kwargs):
    # load the data from the csv file
    data = pd.read_csv(training_data, index_col=0)
    print(f"Dataset shape is {data.values.shape}")
    # split the data into train and test sets
    split_index = int(train_ratio * len(data))
    train_data = data.iloc[:split_index, :]
    test_data = data.iloc[split_index:, :]

    # create the datasets and dataloaders
    train_dataset, test_dataset, train_dataloader, test_dataloader = None, None, None, None
    if len(train_data):
        train_dataset = MultivariateTimeSeriesDataset(train_data, seq_len=seq_len,
                                                      normalize=normalize, standardize=standardize,
                                                      differentiate=differentiate, start_zero=start_zero, shuffle=True)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if len(test_data):
        test_dataset = MultivariateTimeSeriesDataset(test_data, seq_len=seq_len,
                                                      normalize=normalize, standardize=standardize,
                                                      differentiate=differentiate, start_zero=start_zero, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader, train_dataset.scaler
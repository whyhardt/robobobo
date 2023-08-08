import torch
import torch.nn as nn
from typing import Optional
import numpy as np

from utils.get_filter import moving_average


class DataProcessor:
    """Class for processing stock data for the agent.
    Applies Filter, Downsampling, Autoencoder and Predictions to the data given as the stock prices.
    Multiple types of predictions can be applied.
    Short-term predictions take the data as it is and predict the next n steps.
    Mid-term predictions take downsampled and filtered data to cover a longer time period and predict the next m steps.
    For each prediction ptype, the data is processed in the following way:
    1. Short-term or mid-term Filter
    2. Short-term or mid-term Downsampling
    3. Short-term or mid-term Autoencoder
    4. Short-term or mid-term Prediction"""

    def __init__(self, scaler=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = {}
        # example for one processor of ptype "ptype":
        # {"ptype": {"filter": {"a": None, "b": None},
        #          "scaler": None,
        #          "downsampling_rate": None,
        #          "autoencoder": None,
        #          "prediction": None, }, }

    def add_processor(self, ptype: str,
                      predictor: Optional[nn.Module]=None,
                      autoencoder: Optional[nn.Module]=None,
                      mvg_avg: Optional[int]=None,
                      downsampling_rate: Optional[int]=None,
                      scaler=None,
                      differentiate=False):
        # check if filter is valid
        # if filter:
        #     if len(filter) != 2:
        #         raise ValueError("Filter must be a tuple with 2 elements, where the first element is "
        #                          "the numerator (a) and the second element is the denominator (b) "
        #                          "for the Butterworth filter from scikit.signal.filtfilt.")

        # add ptype to keys of processor
        self.processor[ptype] = {"differentiate": differentiate,
                                 "mvg_avg": None,
                                 "downsampling_rate": None,
                                 "autoencoder": None,
                                 "predictor": None,
                                 "scaler": None,}
        if differentiate:
            self.processor[ptype]["differentiate"] = differentiate
        if mvg_avg:
            self.processor[ptype]["mvg_avg"] = mvg_avg
        if downsampling_rate:
            self.processor[ptype]["downsampling_rate"] = downsampling_rate
        if scaler:
            self.processor[ptype]["scaler"] = scaler
        if autoencoder:
            self.processor[ptype]["autoencoder"] = autoencoder
        if predictor:
            self.processor[ptype]["predictor"] = predictor

    def process(self, stock_prices: np.ndarray, ptype: str, flatten=False, mask=None):
        """process stock prices
        :param stock_prices: (tensor) stock prices with shape (observation_length, features)
        :param ptype: (str) type of prediction to apply"""

        # check if ptype is valid
        if ptype not in self.processor.keys():
            raise ValueError("ptype must be one of {}".format(tuple(self.processor.keys())))

        masked_values = None
        if mask is not None and (isinstance(mask, int) or isinstance(mask, float) or isinstance(mask, np.ndarray)):
            # mask stock prices
            stock_prices, masked_values = self._mask(stock_prices, mask)

        # differentiate stock prices
        if self.processor[ptype]["differentiate"]:
            stock_prices = self._differentiate(stock_prices)

        # standardize stock prices
        if self.processor[ptype]["scaler"]:
            stock_prices = self._standardize(stock_prices, ptype)

        # filter stock prices
        if self.processor[ptype]["mvg_avg"] is not None:
            stock_prices = self._filter(stock_prices, ptype)

        # encode stock prices
        if self.processor[ptype]["autoencoder"]:
            print('Tried to use encoder in data processor. Currently not implemented.')
            # stock_prices = self._encode(stock_prices, ptype)

        # concatenate masked values and stock prices
        if masked_values is not None:
            # if not isinstance(stock_prices, torch.Tensor):
            #     stock_prices = torch.from_numpy(stock_prices).to(self.device)
            masked_values = np.zeros((len(masked_values), stock_prices.shape[-1])) + mask
            stock_prices = np.concatenate((masked_values, stock_prices), axis=0)  # TODO: only front padding; Make dependent on indeces masked_values

        # downsample stock prices
        if self.processor[ptype]["downsampling_rate"]:
            stock_prices = self._downsample(stock_prices, ptype)

        # predict stock prices
        if self.processor[ptype]["predictor"]:
            stock_prices = self._predict(stock_prices, ptype)
            # decoded = self.processor[ptype]["autoencoder"].decode(stock_prices)
        # check if necessary to decode stock prices
        else:
            stock_prices = torch.tensor(stock_prices, dtype=torch.float32)

        # reshape stock prices
        if flatten:
            stock_prices = stock_prices.reshape(-1, stock_prices.shape[-1] * stock_prices.shape[-2])

        return stock_prices

    def _differentiate(self, stock_prices):
        """differentiate stock prices
        :param stock_prices: (tensor) stock prices with shape (observation_length, features)"""
        stock_prices = np.diff(stock_prices, axis=0)
        stock_prices = np.concatenate((stock_prices[0, :].reshape(1, -1), stock_prices), axis=0)
        return stock_prices

    def _standardize(self, stock_prices, ptype):
        return self.processor[ptype]["scaler"].transform(stock_prices)

    def _filter(self, stock_prices, ptype):
        win_len = np.min((len(stock_prices), self.processor[ptype]["mvg_avg"]))
        return moving_average(stock_prices, win_len)

    def _downsample(self, stock_prices, ptype):
        return stock_prices[::-self.processor[ptype]["downsampling_rate"]][::-1]

    def _encode(self, stock_prices, ptype):
        # check if stock_prices is a tensor
        if not isinstance(stock_prices, torch.Tensor):
            stock_prices = torch.tensor(stock_prices, dtype=torch.float32).to(self.device)
        return self.processor[ptype]["autoencoder"].encode(stock_prices)

    def _predict(self, stock_prices, ptype):
        # check if stock_prices is a tensor
        if not isinstance(stock_prices, torch.Tensor):
            stock_prices = torch.tensor(stock_prices, dtype=torch.float32).to(self.device)
        # flatten stock_prices
        # stock_prices = stock_prices.reshape(-1, stock_prices.shape[-1] * stock_prices.shape[-2])
        if self.processor[ptype]["predictor"].latent_dim != stock_prices.shape[-1]:
            # draw latent variable from normal distribution and concatenate it to stock_prices
            latent_variable = torch.randn((stock_prices.shape[0], self.processor[ptype]["predictor"].latent_dim - stock_prices.shape[-1])).to(self.device)
            stock_prices = torch.cat((latent_variable, stock_prices), dim=1).unsqueeze(0).float()
        return self.processor[ptype]["predictor"](stock_prices).squeeze(0)  # .squeeze(2).permute(1, 0)

    def _decode(self, stock_prices, ptype):
        # check if stock_prices is a tensor
        if not isinstance(stock_prices, torch.Tensor):
            stock_prices = torch.tensor(stock_prices, dtype=torch.float32).to(self.device)
        return self.processor[ptype]["autoencoder"].decode(stock_prices)

    def _mask(self, stock_prices, mask):
        # mask stock prices if all values of one time step are 'mask'
        masked_values = np.where(np.all(stock_prices == mask, axis=1))[0]
        stock_prices = np.delete(stock_prices, masked_values, axis=0)
        return stock_prices, masked_values
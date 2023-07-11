# backtesting a trained agent in a verified backtesting environment

from backtesting import Backtest, Strategy


class Agent(Strategy):
    def __init__(self, model):
        self.model = model

    # def next(self):
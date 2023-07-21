# download stock prices from yahoo finance
import os.path

import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import pyplot as plt

# import investpy


def preprocess_stock_data(data: pd.DataFrame, mask_nan=True):
	# interpolate missing values
	data = data.interpolate(limit_area='inside')
	if mask_nan:
		data = data.fillna(0)
	# if only one value in the beginning is nan, replace it with the next value
	for s in data.columns:
		if data[s][0] == 0 and data[s][1] != 0:
			data[s][0] = data[s][1]
		if data[s][-1] == 0 and data[s][-2] != 0:
			data[s][-1] = data[s][-2]
	return data


if __name__ == '__main__':
	save_single_file = False
	max_data_range = True
	mask_nan = True
	dir_base = r'..\stock_data'
	dir_single_files = 'portfolio_custom140'
	file_all_stocks = 'portfolio_custom140_2008_2022_max_range.csv'

	if not os.path.isdir(os.path.join(dir_base, dir_single_files)):
		os.mkdir(os.path.join(dir_base, dir_single_files))

	# list of stocks to download
	steady_stocks = ["PG", "KO", "JNJ", "WMT", "MCD", "VZ", "PEP", "GE", "IBM", "CL", "NKE", "XOM", "CVX", "DIS", "HD",
					 "MMM", "PFE", "CSCO", "INTC", "CAT", "BA", "AXP", "GS", "JPM", "AAPL", "MSFT", "V", "RTX", "T",
					 "FDX", "UPS", "UNH", "AMGN", "ABT", "GILD", "CRM", "ORCL", "ABBV", "TGT", "COST", "KO", "PEP",
					 "PM", "PG", "CHD", "EL", "CLX", "KMB", "GIS", "WMT", "JNJ", "MMM", "HD", "VZ", "DIS", "UNH", "XOM",
					 "CVX", "NKE", "MCD", "IBM", "CAT", "BA", "JPM", "GS", "AAPL", "MSFT", "V", "AXP", "INTC", "CSCO",
					 "RTX", "T", "FDX", "UPS", "AMGN", "ABT", "GILD", "CRM", "ORCL", "TGT", "COST", "KO", "PEP", "PM",
					 "PG", "CHD", "EL", "CLX", "KMB", "GIS"]
	dynamic_stocks = [
		"TSLA", "AAPL", "AMZN", "GOOGL", "META", "NFLX", "NVDA", "MSFT", "ADBE", "PYPL",
		"SHOP", "CRM", "UBER", "ROKU", "DOCU", "PTON", "ZM", "TWLO", "OKTA", "SNAP",
		"CRSP", "PINS", "FSLR", "SEDG", "ZS", "TTD", "WORK", "MDB", "NET", "LULU",
		"NIO", "CRWD", "DDOG", "AVGO", "AMD", "PLTR", "SNOW", "ABNB", "RBLX", "WISH",
		"UPST", "COIN", "DASH", "AI", "AMWL", "OPEN", "BB", "GME", "CLOV", "NOK",
		"BBBY", "AAL", "RKT", "SPCE", "KOSS", "EXPR", "MVIS", "MARA", "RIOT", "TLRY",
		"XPEV", "QS", "CCIV", "APHA", "TLT", "GLD", "TQQQ", "ARKK", "ARKG", "ARKW",
		"ARKF", "QQQ", "SPY", "AAPL", "GOOGL", "MSFT", "AMZN", "FB", "TSLA", "NVDA",
		"PYPL", "ADBE", "CMG", "NFLX", "AMD", "INTC", "QCOM", "CSCO", "MU", "LRCX",
		"TXN", "SBUX", "AVGO", "XOM", "V", "JPM", "MA", "GS", "BAC", "WFC"
	]
	stocks = steady_stocks + dynamic_stocks

	# remove duplicates and sort
	stocks = list(set(stocks))
	stocks.sort()

	# forex and crypto pairs
	forex_pairs = ["EUR=X", "JPY=X", "GBP=X", "CHF=X", "AUD=X", "CAD=X"]
	cryptocurrencies = ["BTC-USD", "ETH-USD", "XRP-USD", "LTC-USD", "BCH-USD", "ADA-USD", "DOGE-USD", "LINK-USD",
						"XLM-USD", "ATOM-USD", "AVAX-USD", "DOT-USD", "ALGO-USD", "VET-USD", "XTZ-USD",
						"SOL-USD", "AAVE-USD", "MKR-USD"]
	forex_pairs = list(set(forex_pairs))
	cryptocurrencies = list(set(cryptocurrencies))
	forex_pairs.sort()
	cryptocurrencies.sort()

	stocks += forex_pairs + cryptocurrencies
	# stocks = cryptocurrencies

	# define start and end date for data download
	# yf_start = pd.Timestamp('2008-01-01')
	# yf_end = pd.Timestamp('2022-12-31')
	yf_start = '2008-01-01'
	yf_end = '2022-12-31'

	# download stocks given in list; remove if not found for given time period
	stock_data = []
	stocks_found = []
	for s in stocks:
		print(f"Downloading {s}...")
		# data = yf.download(s, start='2023-04-11', end='2023-04-18', interval="1m", auto_adjust=True)
		yf_data = yf.download(s, start=yf_start, end=yf_end, auto_adjust=True)#['Open']
		if not save_single_file:
			yf_data = yf_data['Open']

		len_yf_data = len(yf_data['Open']) if isinstance(yf_data, pd.DataFrame) else len(yf_data)
		if len_yf_data > 0:
			stock_data.append(yf_data)
			stocks_found.append(s)

	stocks = stocks_found

	# check if stocks are list of dataframes (OHLC) or list of series (only Open)
	if not isinstance(stock_data[0], pd.DataFrame):
		# list of series
		stock_data = pd.concat(stock_data, axis=1, keys=stocks)
		if max_data_range:
			stock_data = stock_data.reindex(pd.date_range(start=yf_start, end=yf_end, freq='D'))
		stock_data = preprocess_stock_data(stock_data)

		# save dataframes in csv files
		stock_data.to_csv(os.path.join(dir_base, 'portfolio_custom145_2008_2022.csv'))
		print(f"Downloaded {len(stocks)} stocks! Saved them to directory '..\stock_data'.")
	else:
		# list of dataframes
		for i, df in enumerate(stock_data):
			# set index axis to maximum length
			if max_data_range:
				df = df.reindex(pd.date_range(start=yf_start, end=yf_end, freq='D'))
			df = preprocess_stock_data(df, mask_nan=mask_nan)

			# save yahoo stock in csv file
			df.to_csv(os.path.join(dir_base, dir_single_files, f'{stocks[i]}.csv'))
		print(f"Downloaded {len(stocks)} stocks! Saved them to directory '{dir_single_files}'.")


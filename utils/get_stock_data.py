# download stock prices from yahoo finance

import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import pyplot as plt

# import investpy


if __name__ == '__main__':
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
	stocks = list(set(stocks))
	stocks.sort()

	forex_pairs = ["EUR=X", "JPY=X", "GBP=X", "CHF=X", "AUD=X", "CAD=X"]
	cryptocurrencies = ["BTC-USD", "ETH-USD", "XRP-USD", "LTC-USD", "BCH-USD", "ADA-USD", "DOGE-USD", "LINK-USD",
						"XLM-USD", "ATOM-USD", "AVAX-USD", "DOT-USD", "ALGO-USD", "VET-USD", "XTZ-USD",
						"SOL-USD", "AAVE-USD", "MKR-USD"]
	forex_pairs = list(set(forex_pairs))
	cryptocurrencies = list(set(cryptocurrencies))
	forex_pairs.sort()
	cryptocurrencies.sort()

	stocks += forex_pairs + cryptocurrencies

	# define start and end date for data download
	yf_start = '2008-01-01'
	yf_end = '2022-12-31'

	# download stocks given in list; remove if not found for given time period
	stock_data = []
	for s in stocks:
		print(f"Downloading {s}...")
		# data = yf.download(s, start='2023-04-11', end='2023-04-18', interval="1m", auto_adjust=True)
		yf_data = yf.download(s, start=yf_start, end=yf_end, auto_adjust=True)['Open']
		if len(yf_data) == 0:
			print(f"Stock {s} not found!")
			stocks.remove(s)
		else:
			stock_data.append(yf_data)
	df = pd.concat(stock_data, axis=1, keys=stocks)

	# interpolate missing values
	df = df.interpolate(limit_area='inside').fillna(0)
	# if only one value in the beginning is nan, replace it with the next value
	for s in df.columns:
		if df[s][0] == 0 and df[s][1] != 0:
			df[s][0] = df[s][1]

	# find stocks with most zeros
	# df_zeros = df[df == 0].count(axis=0)
	# df_zeros = df_zeros.sort_values(ascending=False)
	# remove 2 stocks with most zeros
	# df = df.drop(columns=[df_zeros.index[0], df_zeros.index[1]])

	# save dataframes in csv files
	# df.to_csv('..\stock_data\portfolio_custom140_2008_2022.csv')
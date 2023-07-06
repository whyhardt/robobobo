# download stock prices from yahoo finance

import numpy as np
import pandas as pd
import yfinance as yf
# import investpy


if __name__ == '__main__':
	# stocks = ['MMM','ABT','ABBV','ACN','ATVI','AYI','ADBE','AMD','AAP','AES','AET',
	# 		'AMG','AFL','A','APD','AKAM','ALK','ALB','ARE','ALXN','ALGN','ALLE',
	# 		'AGN','ADS','LNT','ALL','GOOGL','GOOG','MO','AMZN','AEE','AAL','AEP',
	# 		'AXP','AIG','AMT','AWK','AMP','ABC','AME','AMGN','APH','APC','ADI','ANDV',
	# 		'ANSS','ANTM','AON','AOS','APA','AIV','AAPL','AMAT','APTV','ADM','ARNC',
	# 		'AJG','AIZ','T','ADSK','ADP','AZO','AVB','AVY','BHGE','BLL','BAC','BK',
	# 		'BAX','BBT','BDX','BRK.B','BBY','BIIB','BLK','HRB','BA','BWA','BXP','BSX',
	# 		'BHF','BMY','AVGO','BF.B','CHRW','CA','COG','CDNS','CPB','COF','CAH','CBOE',
	# 		'KMX','CCL','CAT','CBG','CBS','CELG','CNC','CNP','CTL','CERN','CF','SCHW',
	# 		'CHTR','CHK','CVX','CMG','CB','CHD','CI','XEC','CINF','CTAS','CSCO','C','CFG',
	# 		'CTXS','CLX','CME','CMS','KO','CTSH','CL','CMCSA','CMA','CAG','CXO','COP',
	# 		'ED','STZ','COO','GLW','COST','COTY','CCI','CSRA','CSX','CMI','CVS','DHI',
	# 		'DHR','DRI','DVA','DE','DAL','XRAY','DVN','DLR','DFS','DISCA','DISCK','DISH',
	# 		'DG','DLTR','D','DOV','DWDP','DPS','DTE','DRE','DUK','DXC','ETFC','EMN','ETN',
	# 		'EBAY','ECL','EIX','EW','EA','EMR','ETR','EVHC','EOG','EQT','EFX','EQIX','EQR',
	# 		'ESS','EL','ES','RE','EXC','EXPE','EXPD','ESRX','EXR','XOM','FFIV','FB','FAST',
	# 		'FRT','FDX','FIS','FITB','FE','FISV','FLIR','FLS','FLR','FMC','FL','F','FTV',
	# 		'FBHS','BEN','FCX','GPS','GRMN','IT','GD','GE','GGP','GIS','GM','GPC','GILD',
	# 		'GPN','GS','GT','GWW','HAL','HBI','HOG','HRS','HIG','HAS','HCA','HCP','HP','HSIC',
	# 		'HSY','HES','HPE','HLT','HOLX','HD','HON','HRL','HST','HPQ','HUM','HBAN','HII',
	# 		'IDXX','INFO','ITW','ILMN','IR','INTC','ICE','IBM','INCY','IP','IPG','IFF','INTU',
	# 		'ISRG','IVZ','IQV','IRM','JEC','JBHT','SJM','JNJ','JCI','JPM','JNPR','KSU','K','KEY',
	# 		'KMB','KIM','KMI','KLAC','KSS','KHC','KR','LB','LLL','LH','LRCX','LEG','LEN','LUK',
	# 		'LLY','LNC','LKQ','LMT','L','LOW','LYB','MTB','MAC','M','MRO','MPC','MAR','MMC','MLM',
	# 		'MAS','MA','MAT','MKC','MCD','MCK','MDT','MRK','MET','MTD','MGM','KORS','MCHP','MU',
	# 		'MSFT','MAA','MHK','TAP','MDLZ','MON','MNST','MCO','MS','MOS','MSI','MYL','NDAQ',
	# 		'NOV','NAVI','NTAP','NFLX','NWL','NFX','NEM','NWSA','NWS','NEE','NLSN','NKE','NI',
	# 		'NBL','JWN','NSC','NTRS','NOC','NCLH','NRG','NUE','NVDA','ORLY','OXY','OMC','OKE',
	# 		'ORCL','PCAR','PKG','PH','PDCO','PAYX','PYPL','PNR','PBCT','PEP','PKI','PRGO','PFE',
	# 		'PCG','PM','PSX','PNW','PXD','PNC','RL','PPG','PPL','PX','PCLN','PFG','PG','PGR',
	# 		'PLD','PRU','PEG','PSA','PHM','PVH','QRVO','PWR','QCOM','DGX','RRC','RJF','RTN','O',
	# 		'RHT','REG','REGN','RF','RSG','RMD','RHI','ROK','COL','ROP','ROST','RCL','CRM','SBAC',
	# 		'SCG','SLB','SNI','STX','SEE','SRE','SHW','SIG','SPG','SWKS','SLG','SNA','SO','LUV',
	# 		'SPGI','SWK','SBUX','STT','SRCL','SYK','STI','SYMC','SYF','SNPS','SYY','TROW','TPR',
	# 		'TGT','TEL','FTI','TXN','TXT','TMO','TIF','TWX','TJX','TMK','TSS','TSCO','TDG','TRV',
	# 		'TRIP','FOXA','FOX','TSN','UDR','ULTA','USB','UAA','UA','UNP','UAL','UNH','UPS','URI',
	# 		'UTX','UHS','UNM','VFC','VLO','VAR','VTR','VRSN','VRSK','VZ','VRTX','VIAB','V','VNO',
	# 		'VMC','WMT','WBA','DIS','WM','WAT','WEC','WFC','HCN','WDC','WU','WRK','WY','WHR','WMB',
	# 		'WLTW','WYN','WYNN','XEL','XRX','XLNX','XL','XYL','YUM','ZBH','ZION','ZTS']
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
						"XLM-USD", "ATOM-USD", "UNI-USD", "AVAX-USD", "DOT-USD", "ALGO-USD", "VET-USD", "XTZ-USD",
						"SOL-USD", "AAVE-USD", "COMP-USD", "MKR-USD"]
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

	# save dataframes in csv files
	# df.to_csv('..\stock_data\portfolio_custom142_2008_2022.csv')
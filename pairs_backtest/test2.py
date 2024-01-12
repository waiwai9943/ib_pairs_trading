#conda activate backtest
import datetime as dt #in-built module
import yfinance as yf
yf.pdr_override()
from pandas_datareader import data as pdr
from yahoo_fin import stock_info as si
from tqdm import tqdm
import statsmodels.tsa.stattools as ts
import numpy as np
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
yf.pdr_override()
tickers = si.tickers_sp500()
print(tickers)

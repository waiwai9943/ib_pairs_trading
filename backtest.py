import datetime as dt #in-built module
import pandas as pd
from pandas_datareader import data
import yfinance as yf
from yahoo_fin import stock_info as si
import matplotlib.pyplot as plt
from matplotlib import style
from backtesting import Strategy, Backtest
from backtesting.lib import crossover, SignalStrategy,TrailingStrategy
from backtesting.test import GOOG, SMA
from pathlib import Path
import os
from tqdm import tqdm   


def fill_df(df, col):
    df = df[['%s'%col,'numUnits']]
    df['High']= df['%s'%col]
    df['Low']= df['%s'%col]
    df['Open']= df['%s'%col]
    df['Close'] = df['%s'%col]
    return df

def get_signal(df):
    return(df['numUnits'])

df = pd.read_csv('log/KVUE_TFX.csv')

df = fill_df(df,'y')
print(df)

class pairs_trading(Strategy):
    last_sinal = 0
    def init(self):
        self.signal = self.I(get_signal, self.data)
    
    def next(self):
        if not self.position:
            if self.signal[-1] == 1:
                self.last_sinal = 1
                #long y
                self.buy()
            elif self.signal[-1] == -1:
                self.last_sinal = -1
                #short y
                self.sell()
        elif self.position:
            if self.signal[-1] == 0:
                self.position.close()
            


bt = Backtest(df, pairs_trading, cash=10000 ,hedging = True , commission=.002,
    trade_on_close=True, 
    exclusive_orders=True)
stat = bt.run()
print (stat)

bt.plot(plot_volume= False, plot_pl=True)
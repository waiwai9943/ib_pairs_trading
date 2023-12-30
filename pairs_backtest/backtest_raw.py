#Backtesting the result
import datetime as dt #in-built module
import yfinance as yf
yf.pdr_override()
from pandas_datareader import data as pdr
from yahoo_fin import stock_info as si
from tqdm import tqdm
import statsmodels.tsa.stattools as ts
import numpy as np
#Backtest for each pair
import pandas as pd
# importing matplotlib module
import matplotlib.pyplot as plt
plt.style.use('default')
from sklearn.linear_model import LinearRegression
from pairs_backtest import find_pairs
import os


def check_exists (*arg):
    pair = arg[0]
    first_instrument= pair[0]
    second_instrument = pair[1]
    date = dt.datetime.now()
    date_ymd ='%s_%s_%s'%(date.year,date.month,date.day)
    path = 'ln_data_P0.01_US_%s/%s vs %s_%s-%s.csv'%(date_ymd,first_instrument,second_instrument,date.day,date.month)
    return os.path.isfile(path)

def backtest(pairs,backtest_day,signal_rooling_window):
    date = dt.datetime.now()
    date ='%s_%s_%s'%(date.year,date.month,date.day)

    #all_result = pd.DataFrame(columns = ["Pair", "Buy&Hold_cum_return", "Strategy_cum_return"])
    lookback_window = signal_rooling_window

    start = dt.datetime.now()- dt.timedelta(days=backtest_day)
    end = dt.datetime.now()

    changing_pairs = pd.DataFrame(columns =['first_stock','second_stock','from_date','from_signal','To_date','To_signal'])

    for pair in tqdm(pairs): 
        
        if check_exists(pair):
            print('pair ',pair,'exists')
            continue

        first_instrument = pair[0]
        second_instrument = pair[1]

        df_1 = yf.Ticker(first_instrument)
        df_1=df_1.history(start = start, end = end)
        df_2 = yf.Ticker(second_instrument)
        df_2=df_2.history(start = start, end = end)

        #df = pd.read_csv('raw_data_us_%s-%s.csv'%(end.day,end.month))
        #df_1 = df[first_instrument]
        #df_2 = df[first_instrument]
    
        entry_threshold = 2.0
        exit_threshold = 0.15
        stoploss_threshold = 2.3

        if len(df_1) == len(df_2):
          #check if the length are same
          
          #try:
          beta_list,intercept_list = find_pairs.rolling_LS_beta(df_1,df_2,lookback_window)

          spread = (np.log(df_2['Close']) - ( np.multiply(beta_list, np.log(df_1['Close'])))).to_frame('spread')   ###use natural log to calculate the spread; spread = ln(y) - (ln(x)m))

          #try:
          #  is_stationary = stationary_test(spread.drop)                                              ###test stationary
          #  if not is_stationary:
          #    print('ADF test failed')
          #    continue
          #    continue
          #except:
          #    continue
          spread['smooth_zscore'] = find_pairs.smooth_zscore(spread = spread['spread'],lookback_window = lookback_window)
          df =  pd.concat([spread['smooth_zscore']], axis=1)
          df['X_Close']=df_1['Close']
          df['Y_Close']=df_2['Close']
          df['beta'] = beta_list
          #df = df.insert(1, 'beta',  beta)
          #df= rolling_P_value(df,50)
          #print(df)
          df.dropna()
          # Define the entry and exit thresholds for trading signals

          # Initialize trading variables
          position = 0  # 1 for long, -1 for short, 0 for neutral
          positions = []  # To store the trading positions
          # Implement the pair trading strategy
          #Spread = ln(y) - beta * ln(x)
          for i in range(len(df)):
              if ( (df['smooth_zscore'][i] > entry_threshold) and (df['smooth_zscore'][i] < stoploss_threshold))  and position == 0:
                  # If the z-score exceeds the entry threshold and no position is currently open, go short the spread
                  # short y; long x
                  position = -1
                  positions.append(-1)
              elif ( (df['smooth_zscore'][i] < -entry_threshold) and (df['smooth_zscore'][i] > -stoploss_threshold) ) and position == 0:
                  # If the z-score falls below the negative entry threshold and no position is currently open, go long the spread
                  # long y; short x
                  position = 1
                  positions.append(1)  # Long asset1 and short asset2
              elif ((abs(df['smooth_zscore'][i]) < exit_threshold) or abs(df['smooth_zscore'][i]) > stoploss_threshold) and position != 0:
                  # If the z-score falls within the exit threshold and a position is currently open, close the position
                  position = 0
                  positions.append(0)  # Close the position
              else:
                  positions.append(position)  # No action
          #except:
          # print('some errors appear, pass it')
          # continue

          # Backtest the pair trading strategy
          returns = pd.DataFrame(index=df.index)
          returns['Asset1'] = np.log(df['X_Close'] / df['X_Close'] .shift(1))
          returns['Asset2'] = np.log(df['Y_Close']  / df['Y_Close'] .shift(1))
          #returns['Strategy'] = np.multiply(positions, returns['Asset2']-beta.round() *returns['Asset1'])
          returns['Strategy'] = np.multiply(positions,  (returns['Asset2']-returns['Asset1']).shift(-1))
          returns['positions'] = positions


          #find_changing_pairs:
          changing_pairs = changing_pairs.append(find_pairs.changing_pairs_dict(first_instrument,second_instrument,df = returns, beta = beta_list), ignore_index = True)


          #print(returns)
          # Calculate the cumulative returns of the strategy
          cumulative_returns = returns.cumsum()
          cumulative_returns.rename (columns = {'Asset1':'Asset1_sum','Asset2':'Asset2_sum','Strategy':'Strategy_sum'}, inplace = True)
          cumulative_returns['Buy and Hold'] = (np.log(df['X_Close'] / df['X_Close'].shift(1)) + np.log( df['Y_Close'] /  df['Y_Close'].shift(1))).cumsum()
          #print (spread['spread'])
          # Plot the cumulative returns
          #print('cum:',cumulative_returns)
          #print('spread:',spread)
          raw = pd.concat([df,returns,spread['spread'],cumulative_returns[['Asset1_sum','Asset2_sum','Strategy_sum','Buy and Hold']]], axis=1)


          raw['X_sell_price'] = raw[raw['positions']==1]['X_Close']
          raw['Y_Buy_price'] = raw[raw['positions']==1]['Y_Close']
          raw['X_Buy_price'] = raw[raw['positions']==-1]['X_Close']
          raw['Y_sell_price'] =raw[raw['positions']==-1]['Y_Close']

          raw.to_csv('ln_data_P0.01_US_%s/%s vs %s_%s-%s.csv'%(date,first_instrument,second_instrument,end.day,end.month))

          fig, (ax1, ax2, ax3) = plt.subplots(nrows = 3, figsize=(14,20))

          raw[['X_Close','Y_Close']].plot(ax=ax1)
          ax1.scatter(raw.index , raw['Y_Buy_price'] , label = 'Buy' , marker = '^', color = 'green',alpha =1 )
          ax1.scatter(raw.index , raw['X_sell_price'] , label = 'Sell' , marker = 'v', color = 'red',alpha =1 )
          ax1.scatter(raw.index , raw['X_Buy_price'] , label = 'Buy' , marker = '^', color = 'green',alpha =1 )
          ax1.scatter(raw.index , raw['Y_sell_price'] , label = 'Sell' , marker = 'v', color = 'red',alpha =1 )

          cumulative_returns[['Asset1_sum','Asset2_sum','Strategy_sum']].plot(ax = ax2)
          #df['p_value'].plot(ax = ax2)

          spread= spread.fillna(0)
          spread['smooth_zscore'].plot(ax = ax3)

          ax2.axhline(y = 0, color = 'r', linestyle = 'dashed')
          ax3.axhline(y = 2, color = 'r', linestyle = 'dashed')
          ax3.axhline(y = -2, color = 'r', linestyle = 'dashed')
          ax1.grid()
          ax2.grid()
          ax3.grid()
          ax1.legend()

          plt.tight_layout()
          plt.savefig('ln_return_figure_P0.01_US_%s/%s vs %s_%s-%s.jpeg'%(date,first_instrument,second_instrument,end.day,end.month))
        else:
          print('The length of two datafram is not consistent')



    changing_pairs.to_csv('ln_changing_pairs_%s-%s.csv'%(end.day,end.month))
    #all_result.to_csv('ln_all_result_P0.001_%s-%s.csv'%(end.day,end.month))


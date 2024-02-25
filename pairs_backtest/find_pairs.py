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



def save_txt(in_list):
    date = dt.datetime.today().strftime('%Y_%m_%d')
    with open("ln_%s_pairs_cointegration_P0.01.txt"%date,"w") as f:
        for word in in_list:
            f.write(str(word))
            f.write("\n")
        f.close()

def save_csv(in_list):
    date = dt.datetime.today().strftime('%Y_%m_%d')
    df = pd.DataFrame(in_list, columns =['s1', 's2']) 
    df.to_csv("ln_%s_pairs_cointegration_P0.01.csv"%date)


def plot_coint(pvalue_matrix,market):
  import matplotlib.pyplot as plt
  import seaborn
  fig, ax = plt.subplots(figsize=(20,20))
  if market == 'HK':
    seaborn.heatmap(pvalue_matrix, xticklabels=si.tickers_sp500(), yticklabels=si.tickers_sp500(), cmap='RdYlGn_r'
                , mask = (pvalue_matrix >= 0.05)
                )
  elif market =='US':
     seaborn.heatmap(pvalue_matrix, xticklabels=si.tickers_sp500(), yticklabels=si.tickers_sp500(), cmap='RdYlGn_r'
                     , mask = (pvalue_matrix >= 0.05)
                     )


  end = dt.datetime.now()
  fig.savefig('coint_%s-%s.png'%(end.day,end.month))

def is_same_sector(s1,s2):
    s1_sector = yf.Ticker(s1).info['sector']
    s2_sector = yf.Ticker(s2).info['sector']
    if s1_sector == s2_sector:
       return True
    else:
       return False
    
   



def find_cointegrated_pairs(data):
  ##courtesy:
  ##https://github.com/KidQuant/Pairs-Trading-With-Python/blob/master/PairsTrading.ipynb
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    #print (keys)
    pairs = []
    for i in tqdm(range(n)):
        for j in range(i+1, n):
            try:
              S1 = np.log(data[keys[i]])
              S2 = np.log(data[keys[j]])
              result = ts.coint(S1, S2)
              score = result[0]
              pvalue = result[1]
              score_matrix[i, j] = score
              pvalue_matrix[i, j] = pvalue
              #pairs.append((keys[i], keys[j],pvalue))
              #print(keys[i], keys[j],pvalue)
              if pvalue < 0.05:
                pairs.append((keys[ i], keys[j],pvalue))
                if is_same_sector(keys[i],keys[j]):
                  print('{s1} and {s2} are same sector'.format(s1 = keys[i], s2 = keys[j]))
                  pairs.append((keys[i], keys[j],pvalue))
            except:
                #print('something goes wrong, now passing')
                pass
    print('Returning:',pairs)
    return score_matrix, pvalue_matrix, pairs

def get_US_SNP (lookback, save_data = True ):
    print('getting all s&p stocks....')
    yf.pdr_override()
    tickers = si.tickers_sp500()
    start = dt.datetime.now()- dt.timedelta(days=lookback)
    end = dt.datetime.now()
    df = pdr.get_data_yahoo(tickers, start, end)['Adj Close']
    if save_data:
      df.to_csv('US_SNP.csv')
    return df

def get_US_NAS (lookback, save_data = True ):
    print('getting all nasdaq stocks....')
    yf.pdr_override()
    tickers = si.tickers_nasdaq()
    start = dt.datetime.now()- dt.timedelta(days=lookback)
    end = dt.datetime.now()
    df = pdr.get_data_yahoo(tickers, start, end)['Adj Close']
    if save_data:
      df.to_csv('US_NAS.csv')
    return df

def get_HK_allstock (lookback,save_data):
  import requests
  from bs4 import BeautifulSoup
  url = "https://en.wikipedia.org/wiki/Hang_Seng_Index"
  # Send a GET request to the URL
  response = requests.get(url)
  
  # Parse the HTML content using BeautifulSoup
  soup = BeautifulSoup(response.content, "html.parser")
  
  # Find the second table containing the stock list
  tables = soup.find_all("table", class_="wikitable")
  table = tables[2]  # Assuming the second table is the one you want
  
  # Extract the stock names from the table
  stocks = []
  for row in table.find_all("tr")[1:]:  # Skip the header row
      stock_name = row.find("td").text.strip()
      stocks.append(stock_name)
  
  restructured = []
  # Print the stock list
  for stock in stocks:
      loc = stock.find(':')
      num = str(stock[loc+2:])
      if len(num) < 4:
        for i in range(4-len(num)):
          num = '0' + num
      row = num + '.HK'
  
      restructured.append(row)

  start = dt.datetime.now()- dt.timedelta(days=lookback)
  end = dt.datetime.now()
  df = pdr.get_data_yahoo(restructured, start, end)['Close']
  if save_data:
    df.to_csv('raw_data_hk_%s-%s.csv'%(end.day,end.month))
  return df

def changing_pairs_dict(first,second,df, beta):
  if df.positions[-1]!=df.positions[-2]:
    temp_dict = {'first_stock':first,
                'second_stock':second,
                'from_date':'%s-%s-%s'%(df.index[-2].year,df.index[-2].month,df.index[-2].day),
                'from_signal':df.positions[-2],
                'To_date':'%s-%s-%s'%(df.index[-1].year,df.index[-1].month,df.index[-1].day),
                'To_signal':df.positions[-1],
                 'Beta':beta[-1]}
    return temp_dict


def stationary_test(spread):
  adf = ts.adfuller(spread, maxlag=1)
  #print('ADF test statistic: %.02f' % adf[0])
  #for key, value in adf[4].items():
  #    print('\t%s: %.3f' % (key, value))
  print('p-value: %.03f' % adf[1])
  if adf[1] < 0.05:
    return True
  else:
    return False


def rolling_P_value (df,window):
    num_of_rows = len(df)
    p_value_list = []
    for i in range(num_of_rows):
        if i < window:
            p_value_list.append(np.NaN)
        else:
          try:
            p_value = ts.coint(df[i-window:i]['X_Close'],df[i-window:i]['Y_Close'])[1]
            p_value_list.append(p_value)
          except:
            p_value_list.append(np.NaN)
            pass
    df['p_value'] = p_value_list
    return df

def rolling_LS_beta (df_1,df_2,window):
  num_of_rows = len(df_1)
  beta_list = []
  intercept_list = []
  for i in range(num_of_rows):
    if i < window:
      beta_list.append(np.NaN)
      intercept_list.append(np.NaN)
    else:
      #try:
        #print(df_1,df_2)
        model = LinearRegression()
        model.fit(np.log(df_1[i-window:i]['Close'].values.reshape(-1, 1)),np.log(df_2[i-window:i]['Close']))
        beta = model.coef_[0]
        #beta = model.coef_[0].round()
        #.round()
        beta_list.append(beta)
        intercept = model.intercept_
        intercept_list.append(intercept)
      #except:
      #  beta_list.append(np.NaN)
      #  intercept_list.append(np.NaN)
  return (beta_list,intercept_list)
 

def smooth_zscore(spread,lookback_window):
    zscore = (spread.rolling(window=1).mean()-spread.rolling(window=lookback_window).mean())/spread.rolling(window=lookback_window).std()
    return zscore



if __name__ == '__main__':
  backtest_day = 365
  market = 'US'
  #market = 'HK'
  #df = get_HK_allstock(lookback=backtest_day)
  df = get_US_allstock(lookback=backtest_day)
  print(df)
  #score_matrix, pvalue_matrix, pairs = find_cointegrated_pairs(df)
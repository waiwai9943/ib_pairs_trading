import datetime as dt #in-built module
import yfinance as yf
from yahoo_fin import stock_info as si
from tqdm import tqdm
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib
from pykalman import KalmanFilter
from numpy import log, polyfit, sqrt, std, subtract
import matplotlib.pyplot as plt
#plt.style.use('seaborn-darkgrid')
import seaborn as sns
import ffn
import warnings
import os 
#warnings.filterwarnings('ignore')




#--------------------------------------------------------------------------------------

def hurst(ts):
    """Returns the Hurst Exponent of the time series vector ts"""
    # Create the range of lag values
    lags = range(2, 100)

    # Calculate the array of the variances of the lagged differences
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = polyfit(log(lags), log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0]*2.0

#--------------------------------------------------------------------------------------

def adf_test(df):
    #df = pd.DataFrame({'y':y,'x':x})
    #est = sm.OLS(df.y, df.x)
    #est = est.fit()
    #df['hr'] = -est.params[0]
    #df['spread'] = df.y + (df.x * df.hr)
    adf = ts.adfuller(df['spread'].dropna())
    return adf[1]

#--------------------------------------------------------------------------------------

def half_life(spread):
    spread_lag = spread.shift(1)
    spread_lag.iloc[0] = spread_lag.iloc[1]

    spread_ret = spread - spread_lag
    spread_ret.iloc[0] = spread_ret.iloc[1]

    spread_lag2 = sm.add_constant(spread_lag)

    model = sm.OLS(spread_ret,spread_lag2)
    res = model.fit()
    halflife = int(round(-np.log(2) / res.params[1],0))

    if halflife <= 0:
        halflife = 1
    return halflife

#--------------------------------------------------------------------------------------

def KalmanFilterAverage(x):
    # Construct a Kalman filter
    from pykalman import KalmanFilter
    kf = KalmanFilter(transition_matrices = [1],
                      observation_matrices = [1],
                      initial_state_mean = 0,
                      initial_state_covariance = 1,
                      observation_covariance=1,
                      transition_covariance=.01)

    # Use the observed values of the price to get a rolling mean
    state_means, _ = kf.filter(x.values)
    state_means = pd.Series(state_means.flatten(), index=x.index)
    return state_means

#--------------------------------------------------------------------------------------

#  Kalman filter regression
def KalmanFilterRegression(x,y):

    delta = 1e-3
    trans_cov = delta / (1 - delta) * np.eye(2) # How much random walk wiggles
    obs_mat = np.expand_dims(np.vstack([[x], [np.ones(len(x))]]).T, axis=1)

    kf = KalmanFilter(n_dim_obs=1, n_dim_state=2, # y is 1-dimensional, (alpha, beta) is 2-dimensional
                      initial_state_mean=[0,0],
                      initial_state_covariance=np.ones((2, 2)),
                      transition_matrices=np.eye(2),
                      observation_matrices=obs_mat,
                      observation_covariance=2,
                      transition_covariance=trans_cov)

    # Use the observations y to get running estimates and errors for the state parameters
    state_means, state_covs = kf.filter(y.values)
    return state_means

#--------------------------------------------------------------------------------------

def backtest(x, y, stop_loss = False):
    date = dt.datetime.now()
    date ='%s_%s_%s'%(date.year,date.month,date.day)

    # INPUT:
    # x: the price series of contract one
    # y: the price series of contract two

    # OUTPUT:
    # df1['cum rets']: cumulative returns in pandas data frame
    # sharpe: sharpe ratio

    # Run regression to find hedge ratio and then create spread series
    df1 = pd.DataFrame({'y':y,'x':x})
    state_means = KalmanFilterRegression(KalmanFilterAverage(x),KalmanFilterAverage(y))

    df1['hr'] = - state_means[:,0]
    df1['spread'] = df1.y + (df1.x * df1.hr)

    ##############################################################

    halflife = half_life(df1['spread'])

    ##########################################################

    #meanSpread = df1.spread.rolling(window=halflife).mean()
    #stdSpread = df1.spread.rolling(window=halflife).std()
    roolng_window = 34
    meanSpread = df1.spread.rolling(window=roolng_window).mean()
    stdSpread = df1.spread.rolling(window=roolng_window).std()


    df1['zScore'] = round((df1.spread-meanSpread)/stdSpread,3)

    ##############################################################

    entryZscore = 1.5 #1.5
    exitZscore = -0 #-0.2
    exitZscore_stop_loss = 2

    # Set up num units long
    df1['long entry'] = ((df1.zScore < - entryZscore) & ( df1.zScore.shift(1) > - entryZscore))

    if stop_loss:
        df1['long exit'] = (((df1.zScore > - exitZscore) & (df1.zScore.shift(1) < - exitZscore)) | ((df1.zScore < - exitZscore_stop_loss) & (df1.zScore.shift(1) > - exitZscore_stop_loss)))
    else:
        df1['long exit'] = ((df1.zScore > - exitZscore) & (df1.zScore.shift(1) < - exitZscore))


    df1['num units long'] = np.nan
    df1.loc[df1['long entry'],'num units long'] = 1
    df1.loc[df1['long exit'],'num units long'] = 0
    #df1['num units long'][0] = 0
    df1.loc[0,['num units long']] = 0
    df1['num units long'] = df1['num units long'].fillna(method='pad')

    # Set up num units short
    df1['short entry'] = ((df1.zScore >  entryZscore) & ( df1.zScore.shift(1) < entryZscore))

    if stop_loss:
        df1['short exit'] = (((df1.zScore < exitZscore) & (df1.zScore.shift(1) > exitZscore)) | ((df1.zScore >  exitZscore_stop_loss) & (df1.zScore.shift(1) <  exitZscore_stop_loss)))
    else:
        df1['short exit'] = ((df1.zScore < exitZscore) & (df1.zScore.shift(1) > exitZscore)) 

    df1.loc[df1['short entry'],'num units short'] = -1
    df1.loc[df1['short exit'],'num units short'] = 0

    df1.loc[0,['num units short']] = 0
    df1['num units short'] = df1['num units short'].fillna(method='pad')

    df1['numUnits'] = df1['num units long'] + df1['num units short']
    df1['spread pct ch'] = (df1['spread'] - df1['spread'].shift(1)) / ((df1['x'] * abs(df1['hr'])) + df1['y'])
    df1['port rets'] = df1['spread pct ch'] * df1['numUnits'].shift(1)

    df1['cum rets'] = df1['port rets'].cumsum()
    df1['cum rets'] = df1['cum rets'] + 1
 

    #df1['Open'] = (df1['spread'])
    #df1['Low'] = (df1['spread'])
    #df1['High'] = (df1['spread'])
    #df1['Close'] = (df1['spread'])

    try:
        sharpe = ((df1['port rets'].mean() / df1['port rets'].std()) * sqrt(252))
    except ZeroDivisionError:
        sharpe = 0.0

    #start_val = 1
    #end_val = df1['cum rets'].iat[-1]
    #start_date = df1.iloc[0].name
    #end_date = df1.iloc[-1].name
    #days = (end_date - start_date).days
    #CAGR = (end_val / start_val) ** (252.0/days) - 1

    #df1[s1+ " "+s2+'_cum_rets'] = df1['cum rets']
    #############################################################
    #df1.to_csv('kalman_backtest_data_%s/%s_%s_%s.csv'%(date,s1,s2,date))
    return df1, sharpe,halflife



def py_plot(df1,date,first_instrument,second_instrument,path = 'py_plot', is_close = True):
    if not os.path.exists(path):
        os.makedirs(path)

    fig, (ax1, ax2, ax3) = plt.subplots(nrows = 3, figsize=(14,20))
    df1['X_sell_price'] = df1[df1['numUnits']==1]['x']
    df1['Y_Buy_price'] = df1[df1['numUnits']==1]['y']
    df1['X_Buy_price'] = df1[df1['numUnits']==-1]['x']
    df1['Y_sell_price'] =df1[df1['numUnits']==-1]['y']


    ax1 = df1[['x','y']].plot(ax=ax1)
    ax1.scatter(df1.index , df1['Y_Buy_price'] , label = 'Buy' , marker = '^', color = 'green',alpha =1 )
    ax1.scatter(df1.index , df1['X_sell_price'] , label = 'Sell' , marker = 'v', color = 'red',alpha =1 )
    ax1.scatter(df1.index , df1['X_Buy_price'] , label = 'Buy' , marker = '^', color = 'green',alpha =1 )
    ax1.scatter(df1.index , df1['Y_sell_price'] , label = 'Sell' , marker = 'v', color = 'red',alpha =1 )
    ax2 = df1[['cum rets']].plot(ax = ax2)
    #df['p_value'].plot(ax = ax2)
    df1['zScore']= df1['zScore'].fillna(0)
    ax3= df1['zScore'].plot(ax = ax3)
    ax2.axhline(y = 1, color = 'r', linestyle = 'dashed')
    ax3.axhline(y = 1.5, color = 'r', linestyle = 'dashed')
    ax3.axhline(y = -1.5, color = 'r', linestyle = 'dashed')
    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax1.legend()
    plt.tight_layout()
    #plt.show()
    plt.savefig(path+'/%s vs %s_%s.jpeg'%(first_instrument,second_instrument,date))
    if is_close:
        plt.close('all')
    return ax1,ax2,ax3
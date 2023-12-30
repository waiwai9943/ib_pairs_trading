import futu as ft
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as ts
import os 
plt.style.use('default')

class pairs():
    def __init__(self, first_instrument, second_instrument, lookback_period = 365):
        self.pair1 = first_instrument
        self.pair2 = second_instrument
        #self.pair1_FT = number_alignment_FUTU(pair1,Market) 
        #self.pair2_FT = number_alignment_FUTU(pair1,Market) 
        self.start_date = dt.datetime.now()- dt.timedelta(days = lookback_period)
        self.end_date =  dt.datetime.now()
        self.df_pair1 = self.getdata(first_instrument, self.start_date, self.end_date)
        self.df_pair2 = self.getdata(second_instrument, self.start_date, self.end_date)
        self.df_pair1,self.df_pair2 = self.check_date(self.df_pair1,self.df_pair2)
        self.is_coint  = self.check_cointegration(self.df_pair1,self.df_pair2)
        self.hedge_ratio, self.spread = self.spread_calculation (self.df_pair1,self.df_pair2)
        self.is_mean_reverting = self.check_adf(self.spread)
        self.zscore = self.smooth_zscore(self.spread)
        self.df_pair = self.combine_data()
        self.positions = self.run_strategy(entry_threshold = 2, exit_threshold =0.1)
        self.df_return = self.stock_return()
        self.df_cum_return = self.cum_return()
        self.plot_return = self.plot_cum_return()
    

    
    def concat_data(self,  first_data, second_data):
        data = pd.concat(first_data,second_data, axis = 0, verify_integrity= True)
        return data


    def plot_cum_return(self):
        self.df_cum_return.plot()

    def cum_return (self):
        # Calculate the cumulative returns of the strategy
        cumulative_returns = self.df_return.cumsum()
        cumulative_returns['Buy and Hold'] = np.log(self.df_pair['X_Close'] / self.df_pair['X_Close'].shift(1)) + np.log( self.df_pair['Y_Close'] /  self.df_pair['Y_Close'].shift(1)).cumsum()
        return cumulative_returns
    
    def run_strategy(self, entry_threshold = 2, exit_threshold =0.1):
                # Define the entry and exit thresholds for trading signals
        # Initialize trading variables
        position = 0  # 1 for long, -1 for short, 0 for neutral
        positions = []  # To store the trading positions

        # Implement the pair trading strategy
        for i in range(len(self.df_pair)):
            if self.df_pair['zscore'][i] > entry_threshold and position == 0:
                # If the z-score exceeds the entry threshold and no position is currently open, go long the spread
                position = 1
                positions.append(-1)  # Short asset1 and long asset2
            elif self.df_pair['zscore'][i] < -entry_threshold and position == 0:
                # If the z-score falls below the negative entry threshold and no position is currently open, go short the spread
                position = -1
                positions.append(1)  # Long asset1 and short asset2
            elif abs(self.df_pair['zscore'][i]) < exit_threshold and position != 0:
                # If the z-score falls within the exit threshold and a position is currently open, close the position
                position = 0
                positions.append(0)  # Close the position
            else:
                positions.append(position)  # No action
        self.df_pair['positions'] = positions
        self.df_pair['spread'] = self.spread
        self.save_data() ##saving the data
        return positions



    def stock_return (self):
        returns = pd.DataFrame(index=self.df_pair.index)
        returns['Asset1'] = np.log(self.df_pair['X_Close'] / self.df_pair['X_Close'] .shift(1))
        returns['Asset2'] = np.log(self.df_pair['Y_Close']  / self.df_pair['Y_Close'] .shift(1))
        returns['Strategy'] = np.multiply(self.positions, returns['Asset1'] - returns['Asset2'])
        return returns
    
    def combine_data (self):
        df = pd.DataFrame(columns=['X_Close','Y_Close','zscore'])
        df['X_Close']=self.df_pair1['Close']
        df['Y_Close']=self.df_pair2['Close']
        df['zscore']=self.zscore
        return df
    
    
    def smooth_zscore(self, spread, swindow = 50):
        zscore = (spread.rolling(window=1).mean()-spread.rolling(window=swindow).mean())/spread.rolling(window=swindow).std()
        return zscore

    def getdata (self, stock, start_date, end_date):
        columns = ['Open','High','Low','Close','Volume']
        df = yf.Ticker(stock)
        df = df.history(start = start_date, end = end_date)
        df.index = df.index.strftime('%d/%m/%Y')
        df = df[columns]
        return (df)

    def check_date (self, first_df, second_df):
        common_index = (first_df.index).intersection(second_df.index) 
        first_df = first_df.loc[common_index].copy()
        second_df = second_df.loc[common_index].copy()
        return first_df, second_df

    def check_cointegration (self, first_df, second_df):
        score, pvalue, _ = coint(np.log(first_df['Close']), (np.log(second_df['Close'])), trend='c', autolag='BIC')
        if pvalue < 0.01:  #significance level
            print(f' Engel-Granger test p-value: {pvalue}\n Rejected null hypothesis.')
            print('Pased the cointegration test')
            return 1
        else: 
            print('Failed the cointegration test')
            return 0
        
    def spread_calculation (self, first_df, second_df):
        model = LinearRegression()
        model.fit(first_df['Close'].values.reshape((-1, 1)),second_df['Close'])
        beta = (model.coef_).round() #Hedge Ratio 
        spread = (first_df['Close'] - (beta) * second_df['Close']).to_frame('spread')
        return beta,spread
    
    def check_adf (self, spread):
        adf = ts.adfuller(spread, maxlag=1)
        print('ADF test statistic: %.02f' % adf[0])
        for key, value in adf[4].items():
            print('\t%s: %.3f' % (key, value))
        print('p-value: %.03f' % adf[1])
        if adf[1] < 0.01:
            print('Pased the ADF test')
            return 1
        else:
            print('Failed the ADF test')
            return 0
    
    def plot_zscore(self):
        plt.figure(figsize=(16, 8), dpi=150)
        self.zscore.plot()
        plt.title('Spread: %s - %s'%(self.pair1,self.pair2))
        plt.show()

    def save_data(self):
        newpath = 'log/'
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        self.df_pair.to_csv(newpath+'log_%s_%s.csv'%(self.pair1,self.pair2))
        print('Saved the log')


class pair_strategy(object):
    # API parameter setting
    api_svr_ip = '127.0.0.1'  # 账户登录的牛牛客户端PC的IP, 本机默认为127.0.0.1
    api_svr_port = 11111  # 富途牛牛端口，默认为11111
    unlock_password = "345498"  # 美股和港股交易解锁密码
    trade_env = ft.TrdEnv.SIMULATE
    budget = 100000

    def __init__(self, stock1, stock2, data, observation):
        """
        Constructor
        """
        self.stock1 = stock1
        self.stock2 = stock2
        self.dataframe = data
        self.observation = observation
        self.quote_ctx, self.trade_ctx = self.context_setting()

    def close(self):
        self.quote_ctx.close()
        self.trade_ctx.close()

    def context_setting(self):
        """
        API trading and quote context setting
        :returns: trade context, quote context
        """
        if self.unlock_password == "":
            raise Exception("请先配置交易解锁密码! password: {}".format(
                self.unlock_password))

        quote_ctx = ft.OpenQuoteContext(
            host=self.api_svr_ip, port=self.api_svr_port)

        if 'HK.' in self.stock1:
            trade_ctx = ft.OpenHKTradeContext(host=self.api_svr_ip, port=self.api_svr_port)
        elif 'US.' in self.stock1:
            trade_ctx = ft.OpenUSTradeContext(host=self.api_svr_ip, port=self.api_svr_port)
        else:
            raise Exception("不支持的stock: {}".format(self.stock1))

        if self.trade_env == ft.TrdEnv.REAL:
            ret_code, ret_data = trade_ctx.unlock_trade(
                self.unlock_password)
            if ret_code == ft.RET_OK:
                print('解锁交易成功!')
            else:
                raise Exception("请求交易解锁失败: {}".format(ret_data))
        else:
            print('解锁交易成功!')

        return quote_ctx, trade_ctx

    def calculate_position(self,df,stock1,stock2,hedge_ration):
        last_price_stock1 = df.X_Close[-1]
        last_price_stock2 = df.Y_Close[-1]
        ret, stock_data = quote_ctx.get_stock_basicinfo(ft.Market.HK, ft.SecurityType.STOCK, [stock1, stock2])
        if ret == ft.RET_OK:
            min_share_stock1 = stock_data['lot_size'][0]
            min_share_stock2 = stock_data['lot_size'][0]
        else:
            print('error:', stock_data)
        min_capital_stock1 = last_price_stock1*min_share_stock1
        min_capital_stock2 = last_price_stock2*min_share_stock2
        min_capital_total = min_capital_stock1 + hedge_ration * min_capital_stock2
        contract_size = self.budget//min_capital_total
        return min_share_stock1,min_share_stock2,contract_size
        



    def execute(self):
        today = dt.datetime.today()
        pre_day = (today - dt.timedelta(days=self.observation)
                   ).strftime('%Y-%m-%d')
        end_dt = today.strftime('%Y-%m-%d')
        ret_code, prices, page_req_key = self.quote_ctx.request_history_kline(self.stock1, start=pre_day, end=end_dt)
        if ret_code != ft.RET_OK:
            print("request_history_kline fail: {}".format(prices))
            return
        
        #data = self.dataframe
        min_share_stock1, min_share_stock2, contract_size  = self.calculate_position(self.dataframe.df_pair,self.stock1,
                                                                                     self.stock2,self.dataframe.hedge_ratio)
        hedge_ratio = self.dataframe.hedge_ratio
        stock1_share = min_share_stock1*contract_size
        stock2_share = min_share_stock2*contract_size*hedge_ratio

        #check holding
        ret_code, data = self.trade_ctx.position_list_query(
            trd_env=self.trade_env)
        if ret_code != ft.RET_OK:
            raise Exception('账户信息获取失败: {}'.format(data))
        pos_info = data.set_index('code')        

        if (self.stock1 not in data['code']) and (self.stock2 not in data['code']):  #not holding the paris currently

            if pair.df_pair.positions[-1] == -1:
                #short asset1 and long asset2
                ret_code, data = self.trade_ctx.position_list_query(
                    trd_env=self.trade_env)

                if ret_code != ft.RET_OK:
                    raise Exception('账户信息获取失败: {}'.format(data))
                pos_info = data.set_index('code')


                ##Short asset 1
                ret_code, data = self.quote_ctx.get_market_snapshot([self.stock1])
                if ret_code != 0:
                    raise Exception('市场快照数据获取异常 {}'.format(data))
                cur_price = data['last_price'][0]
                ret_code, ret_data = self.trade_ctx.place_order(
                    price= cur_price,
                    qty = stock1_share,
                    code=self.stock1,
                    trd_side=ft.TrdSide.SELL_SHORT,
                    order_type=ft.OrderType.NORMAL,
                    trd_env=self.trade_env)

                if not ret_code:
                    print(
                        ' MAKE BUY ORDER\n\tcode = {} price = {} quantity = {}'
                        .format(self.stock1, cur_price, stock1_share))
                else:
                    print('MAKE BUY ORDER FAILURE: {}'.format(ret_data))


                ##Long asset 2
                ret_code, data = self.quote_ctx.get_market_snapshot([self.stock2])
                if ret_code != 0:
                    raise Exception('市场快照数据获取异常 {}'.format(data))
                cur_price = data['last_price'][0]

                ret_code, ret_data = self.trade_ctx.place_order(
                   price= cur_price,
                   qty = stock2_share,
                   code=self.stock2,
                   trd_side=ft.TrdSide.BUY,
                   order_type=ft.OrderType.NORMAL,
                   trd_env=self.trade_env) 

                if not ret_code:
                    print(
                        ' MAKE BUY ORDER\n\tcode = {} price = {} quantity = {}'
                        .format(self.stock2, cur_price, stock2_share))
                else:
                    print('MAKE BUY ORDER FAILURE: {}'.format(ret_data))


            if pair.df_pair.positions[-1] == 1:
                #long asset1 and short asset2
                ret_code, data = self.trade_ctx.position_list_query(
                    trd_env=self.trade_env)

                if ret_code != ft.RET_OK:
                    raise Exception('账户信息获取失败: {}'.format(data))
                pos_info = data.set_index('code')


                ##Long asset 1
                ret_code, data = self.quote_ctx.get_market_snapshot([self.stock1])
                if ret_code != 0:
                    raise Exception('市场快照数据获取异常 {}'.format(data))
                cur_price = data['last_price'][0]
                ret_code, ret_data = self.trade_ctx.place_order(
                    price= cur_price,
                    qty = stock1_share,
                    code=self.stock1,
                    trd_side=ft.TrdSide.BUY,
                    order_type=ft.OrderType.NORMAL,
                    trd_env=self.trade_env)

                if not ret_code:
                    print(
                        ' MAKE BUY ORDER\n\tcode = {} price = {} quantity = {}'
                        .format(self.stock1, cur_price, stock1_share))
                else:
                    print('MAKE BUY ORDER FAILURE: {}'.format(ret_data))

                ##Short asset 2
                ret_code, data = self.quote_ctx.get_market_snapshot([self.stock2])
                if ret_code != 0:
                    raise Exception('市场快照数据获取异常 {}'.format(data))
                cur_price = data['last_price'][0]

                ret_code, ret_data = self.trade_ctx.place_order(
                   price= cur_price,
                   qty = stock2_share,
                   code=self.stock2,
                   trd_side=ft.TrdSide.SELL_SHORT,
                   order_type=ft.OrderType.NORMAL,
                   trd_env=self.trade_env)         

                if not ret_code:
                    print(
                        ' MAKE BUY ORDER\n\tcode = {} price = {} quantity = {}'
                        .format(self.stock2, cur_price, stock2_share))
                else:
                    print('MAKE BUY ORDER FAILURE: {}'.format(ret_data))
        else:
            ###Close position
            if pair.df_pair.positions[-1] == 0:
                #long asset1 and short asset2
                ret_code, data = self.trade_ctx.position_list_query(
                    trd_env=self.trade_env)

                if ret_code != ft.RET_OK:
                    raise Exception('账户信息获取失败: {}'.format(data))
                pos_info = data.set_index('code')
                stock1_holding = int(pos_info['qty'][self.stock1])
                stock2_holding = int(pos_info['qty'][self.stock2])

            ##close asset 1
            if stock1_holding > 0: ##Currently long in asset 1, want to clsoe it
                ret_code, data = self.quote_ctx.get_market_snapshot([self.stock1])
                if ret_code != 0:
                    raise Exception('市场快照数据获取异常 {}'.format(data))
                cur_price = data['last_price'][0]
                ret_code, ret_data = self.trade_ctx.place_order(
                    price= cur_price,
                    qty = stock1_holding,
                    code=self.stock1,
                    trd_side=ft.TrdSide.SELL,
                    order_type=ft.OrderType.NORMAL,
                    trd_env=self.trade_env)
                
            if stock1_holding < 0: ##Currently short in asset 1, want to clsoe it
                ret_code, data = self.quote_ctx.get_market_snapshot([self.stock1])
                if ret_code != 0:
                    raise Exception('市场快照数据获取异常 {}'.format(data))
                cur_price = data['last_price'][0]
                ret_code, ret_data = self.trade_ctx.place_order(
                    price= cur_price,
                    qty = abs(stock1_holding),
                    code=self.stock1,
                    trd_side=ft.TrdSide.BUY,
                    order_type=ft.OrderType.NORMAL,
                    trd_env=self.trade_env)

            ##close asset 2
            if stock2_holding > 0: ##Currently long in asset 1, want to clsoe it
                ret_code, data = self.quote_ctx.get_market_snapshot([self.stock2])
                if ret_code != 0:
                    raise Exception('市场快照数据获取异常 {}'.format(data))
                cur_price = data['last_price'][0]
                ret_code, ret_data = self.trade_ctx.place_order(
                    price= cur_price,
                    qty = stock2_holding,
                    code=self.stock2,
                    trd_side=ft.TrdSide.SELL,
                    order_type=ft.OrderType.NORMAL,
                    trd_env=self.trade_env)
                
            if stock2_holding < 0: ##Currently short in asset 1, want to clsoe it
                ret_code, data = self.quote_ctx.get_market_snapshot([self.stock1])
                if ret_code != 0:
                    raise Exception('市场快照数据获取异常 {}'.format(data))
                cur_price = data['last_price'][0]
                ret_code, ret_data = self.trade_ctx.place_order(
                    price= cur_price,
                    qty = abs(stock2_holding),
                    code=self.stock2,
                    trd_side=ft.TrdSide.BUY,
                    order_type=ft.OrderType.NORMAL,
                    trd_env=self.trade_env)


def number_alignment_FUTU (num, market):
    row = []
    num = str(num)
    if len(num) < 5:
        for i in range(5-len(num)):
            num = '0' + num
    row =  market + '.' + num 
    return row


        
# Define your pairs and parameters
pair1 = '2007'  # replace with your desired stock symbols
pair2 = '2319'
Market = 'HK'
pair1_YT = pair1 + '.' + Market
pair2_YT = pair2 + '.' + Market
pair1_FT = number_alignment_FUTU(pair1,Market) 
pair2_FT = number_alignment_FUTU(pair1,Market) 


pair = pairs(pair1_YT, pair2_YT, 365)  #API to yfinance
#print (pair.is_coint)
#print (pair.is_mean_reverting)
#print (pair.df_pair1[:5])
#print (pair.df_pair2[:5])
#pair.plot_cum_return()


# 实例化行情上下文对象
quote_ctx = ft.OpenQuoteContext(host="127.0.0.1", port=11111)

# 上下文控制
quote_ctx.start()              # 开启异步数据接收
quote_ctx.set_handler(ft.TickerHandlerBase())  # 设置用于异步处理数据的回调对象(可派生支持自定义)

threshold = 2 # threshold for z-score
end_date =  dt.datetime.now()
end_date = '%s-%s-%s'%(end_date.year,end_date.month,end_date.day)
start_date = dt.datetime.now()- dt.timedelta(days=10)
start_date = '%s-%s-%s'%(start_date.year,start_date.month,start_date.day)

ft_pair = pair_strategy(pair1_FT,pair2_FT,pair,100)
ft_pair.execute()
ft_pair.close()



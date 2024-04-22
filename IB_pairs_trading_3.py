#import pairs_backtest as pb
import datetime as dt #in-built module
import pandas as pd
from pandas_datareader import data
import yfinance as yf
from yahoo_fin import stock_info as si
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import style
from pathlib import Path
import os
import logging
from tqdm import tqdm   
import glob
import datetime as dt
from pandas_datareader import data as pdr
import numpy as np
from pairs_backtest import kalman_lib as kalman
from pairs_backtest import find_pairs
from pairs_backtest import utils
import ib_insync 
ib_insync.util.startLoop()  # only use in interactive environments (i.e. Jupyter Notebooks)


def delete_folder_contents(folder_path):
    # Iterate over all the items in the folder
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)

        # Check if the item is a file
        if os.path.isfile(item_path):
            # Delete the file
            os.remove(item_path)
        # Check if the item is a subfolder
        elif os.path.isdir(item_path):
            # Recursively delete the subfolder and its contents
            delete_folder_contents(item_path)
            # Delete the empty subfolder
            os.rmdir(item_path)
            

class ib_pt ():

    df_list = []
    first_inst = [] 
    second_inst = []

    def __init__(self, backtest_day, len_constrain_day, start_date, end_date, coint_pairs_type,capital, account, no_pairs, connect_to_ib = True):
        self.backtest_day = backtest_day
        self.len_constrain_day = len_constrain_day
        date = dt.datetime.now()
        self.today = '%s_%s_%s'%(date.year,date.month,date.day)
        self.start = start_date
        self.end = end_date
        self.capital = capital
        self.coint_pairs_type = coint_pairs_type
        self.holding_pairs = []
        self.df = []
        self.account = account
        self.required_no_pairs = no_pairs
        self.init()
        if connect_to_ib:
            print('Selected to online mode')
            self.ib = self.run_ib()
        else:
            print('Selected to offline mode')
            self.ib = False


       
    def init(self):
        if not os.path.exists('log'):
            os.makedirs('log')
        if not os.path.exists('pairs_data'):
           os.makedirs('pairs_data')
        if not os.path.exists('Changing'):
           os.makedirs('Changing')
        if not os.path.exists('trades'):
           os.makedirs('trades')  
        #dir = 'py_plot'
        #for f in os.listdir(dir):
        #  os.remove(os.path.join(dir, f))
        if not os.path.exists('trades/holding_pairs.csv'):
            empty_df = pd.DataFrame([],columns = ['first_stock','first_quantity','second_stock','second_quantity','Hedge Ratio'])
            empty_df.to_csv('trades/holding_pairs.csv',index = False, header = True)

    def _logging(self,msg):
        log_file = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")
        logging.basicConfig(filename='logging/'+log_file, level=logging.INFO, format='%(asctime)s %(message)s')
        logging.info("Logging:")
        if isinstance(msg, pd.DataFrame):
            logging.info("\n" + msg.to_string())
        else:
            logging.info("\n" +msg)

    
    def load_existing_pairs_csv (self, type = 'last_result_NAS', head = 1000):
        #date = dt.datetime.today().strftime('%Y_%m_%d')
        if os.path.isfile("pairs_%s.csv"%self.today): #if file exists
          df = pd.read_csv("pairs_%s.csv"%self.today)
          return df
        elif type == 'allUS_S&P': # if file does not exist, find pairs
          print('selected allUS_S&P')
          df = find_pairs.get_US_SNP(lookback=self.backtest_day)
          print('All_US_STOCK_S&P',df)
          score_matrix, pvalue_matrix, pairs = find_pairs.find_cointegrated_pairs(df)
          print (pairs)
          #find_pairs.save_txt(pairs)  #save all the pairs
          df_pairs = pd.DataFrame(pairs, columns = ['s1','s2','p-value'])
          df_pairs = df_pairs.sort_values(by=['p-value']).head(head)
          df_pairs.to_csv("pairs_%s.csv"%self.today)
          return df_pairs
        
        elif type == 'allUS_NAS': # if file does not exist, find pairs
          print('selected allUS_NAS')
          df = find_pairs.get_US_NAS(lookback=self.backtest_day)
          print('All_US_STOCK_NAS',df)
          score_matrix, pvalue_matrix, pairs = find_pairs.find_cointegrated_pairs(df)
          print (pairs)
          #find_pairs.save_txt(pairs)  #save all the pairs
          df_pairs = pd.DataFrame(pairs, columns = ['s1','s2','p-value'])
          df_pairs = df_pairs.sort_values(by=['p-value']).head(head)
          df_pairs.to_csv("pairs_%s.csv"%self.today)
          return df_pairs
          
        elif type == 'last_result_S&P':
            csv_files_date = []
            csv_files = glob.glob(os.path.join(os.getcwd(), 'pairs*.csv')) #get all the csv file (e.g. pairs_yyyy_mm_dd)
            for csv_file in csv_files:
               filename = os.path.split(csv_file)[1]
               date = str(filename[6:-4])
               date = dt.datetime.strptime(date ,'%Y_%m_%d').date()
               csv_files_date.append(date)
            max_date = max(csv_files_date)
            latest_path = os.getcwd()+'\pairs_%s_%s_%s.csv'%(max_date.year,max_date.month,max_date.day)
            print('Latest path of existing pairs which passed cointegration test:',latest_path)
            df = pd.read_csv(latest_path,index_col=0)
            #print (df)
            return df
        
        elif type == 'last_result_NAS':
            csv_files_date = []
            csv_files = glob.glob(os.path.join(os.getcwd(), 'pairs*.csv')) #get all the csv file (e.g. pairs_yyyy_mm_dd)
            for csv_file in csv_files:
               filename = os.path.split(csv_file)[1]
               date = str(filename[6:-7])
               date = dt.datetime.strptime(date ,'%Y_%m_%d').date()
               csv_files_date.append(date)
            max_date = max(csv_files_date)
            latest_path = os.getcwd()+'\pairs_%s_%s_%s_NAS.csv'%(max_date.year,max_date.month,max_date.day)
            print('Latest path of existing pairs which passed cointegration test:',latest_path)
            df = pd.read_csv(latest_path,index_col=0)
            #print (df)
            return df
        else:
           raise Exception("Please specify 'TYPE'") 

    def get_IB_position(self, save_to_csv = True): 
        position = self.ib.positions()
        position_list = []
        for item in position:
            dictionary = {'symbol':item.contract.symbol,
                        'position':item.position,
                        'avgCost':item.avgCost}
            position_list.append(dictionary)

        if save_to_csv:
            pd.DataFrame(position_list).to_csv('trades/local_all_holding.csv', index = False, header = True)
        return position_list
        
    
    def get_local_position(self):
        holding_pairs= pd.read_csv('trades/holding_pairs.csv')
        self.holding_pairs = holding_pairs
        if holding_pairs.empty:
            print('local holding is empty')
            return True , holding_pairs
        else:
            return False , holding_pairs
    
    def get_trades_in_IB(self):
        trades = self.ib.trades()
        return trades

    def load_latest_changing_pairs(self):
        directory = 'Changing'
        csv_files = glob.glob(os.path.join(directory, '*.csv'))
        csv_files_date = []
        #extract the date of the existing directory
        for csv_file in csv_files:
            filename = os.path.split(csv_file)[1]
            date = str(filename[15:-4])
            date = dt.datetime.strptime(date ,'%Y_%m_%d').date()
            csv_files_date.append(date)
    
        max_date = max(csv_files_date)
        # Get the latest CSV file
        latest_path = directory+'\changing_pairs_%s_%s_%s.csv'%(max_date.year,max_date.month,max_date.day)
        print('Latest path:',latest_path)
        df = pd.read_csv(latest_path,index_col=0)
        return df,max_date
    

    def find_changing_pairs(self,len_constrain_day = 365):
        '''
        Return: dataframe of the changing pairs
        '''
        delete_folder_contents('pairs_data')

        pairs = self.load_existing_pairs_csv(type = self.coint_pairs_type)
        df_changing_pairs = pd.DataFrame(columns =['first_stock','second_stock','from_date','from_signal','To_date','To_signal','Hedge Ratio'])
        for i in range(len(pairs)):
            pair = pairs.iloc[i]
            first_instrument = pair['s1']
            second_instrument = pair['s2']

            df_1 = self.get_data(first_instrument,self.start,self.end)
            df_2 = self.get_data(second_instrument,self.start,self.end)
            common_index = (df_1.index).intersection(df_2.index)
            df_1 = df_1.loc[common_index].copy()
            df_2 = df_2.loc[common_index].copy()
            if len(common_index) > len_constrain_day: ##length constraint
                df, sharpe, half_life = kalman.backtest(df_1['Adj Close'], df_2['Adj Close'])
                df = df.dropna()
                df = df[['x','y','hr','zScore','numUnits','cum rets']]
                df.to_csv('pairs_data/%s-%s.csv' % (first_instrument,second_instrument))
                #find_changing_pairs:
                #changing_pairs = changing_pairs.append(utils.changing_pairs_dict(first_instrument,second_instrument,df = df), ignore_index = True)
                changing_pairs = utils.changing_pairs_dict(first_instrument,second_instrument,df = df)
                if changing_pairs is not None:
                    pairs_dict = pd.DataFrame.from_dict(changing_pairs,orient='index').T
                    df_changing_pairs= pd.concat([df_changing_pairs,pairs_dict], ignore_index = True)
                    kalman.py_plot(df,self.today,first_instrument,second_instrument)
        df_changing_pairs.to_csv('Changing/changing_pairs_%s.csv'%(self.today))
        return df_changing_pairs


    def get_data(self,code,start,end):
        #data = yf.Ticker(code)
        #data = data.history(start = start, end = end)
        #data.index = data.index.strftime('%d/%m/%Y')
        df=pdr.get_data_yahoo(code,start,end)
        return df
    
    def run_ib(self):
        id = 1
        flag = True
        ib = ib_insync.IB()
        while flag:
            try:
                ib.connect(host='127.0.0.1', port=4001, clientId=id, account = self.account, timeout = 5.0) #TWS
                #ib.connect(host='127.0.0.1', port=4002, clientId=id, account = self.account, timeout = 5.0) #Gateway
            except:
                ib.disconnect()
                print ("can't connect to IB API, now try to connect to another cllientID")
                id += 1
            else:
                print ("Connected to IB API")
                flag = False
                return ib
    
    def close_ib(self):
        self.ib.disconnect()

    def compute_signal(self, x ,y):
        x_df = self.get_data(x,self.start,self.end)
        y_df = self.get_data(y,self.start,self.end)
        common_index = (x_df.index).intersection(y_df.index)
        if len(common_index) > self.len_constrain_day:
            x_df = x_df.loc[common_index].copy()
            y_df = y_df.loc[common_index].copy() 
            df, sharpe, _ = kalman.backtest(x_df['Adj Close'], y_df['Adj Close']) #df1 = x; df2 = y
            df = df.dropna()
            path = 'log/%s_%s.csv'%(x,y)
            kalman.py_plot(df,self.today,x,y)
            df.to_csv(path)
            return df
        
    def check_current_pairs (self):
        #out of use
        is_empty, self.holding_pairs = self.get_local_position()
        ib_position_list = self.get_IB_position()
        print('now checking current pairs...')
        local_holding_pairs = self.holding_pairs
        local_holding_pairs_first = (local_holding_pairs['first_stock'].values[self.no_of_pairs])
        local_holding_pairs_second = (local_holding_pairs['second_stock'].values[self.no_of_pairs])
        local_holding_pairs = (local_holding_pairs_first,local_holding_pairs_second)
        self.first_inst.append(local_holding_pairs_first)
        self.second_inst.append(local_holding_pairs_second)
        ib_position_list_symbols = []
        ib_position_list_symbols = [i['symbol'] for i in ib_position_list]
        print("Holdin pairs stored in local drive: ",local_holding_pairs,' ; '
              "All holding in IB: ",ib_position_list_symbols)
        if set(local_holding_pairs).issubset(set(ib_position_list_symbols)):
            print('holding pairs stored in local drive found in all holding in IB: passed')
            df = self.compute_signal(local_holding_pairs_first,local_holding_pairs_second)
            self.df_list.append(df)
            changing_dict = utils.changing_pairs_dict(local_holding_pairs_first,local_holding_pairs_second,df = df)
            print (changing_dict)
            if changing_dict is not None: #Changed signal
                if changing_dict['To_signal'] == 0 or df['numUnits'][-1] == 0:
                  print ('ib_position_list', ib_position_list_symbols)
                  for stock in ib_position_list:
                    if stock['symbol'] in local_holding_pairs:
                        print ('is in local_holding_pairs')
                        temp_contract = ib_insync.Stock(stock['symbol'],exchange='SMART',currency='USD')
                        self.ib.qualifyContracts(temp_contract)
                        print('Closing %s with %s'%(stock['symbol'],stock['position']))
                        if stock['position'] > 0: 
                            temp_order = ib_insync.MarketOrder('SELL', stock['position'])
                            self.ib.placeOrder(temp_contract, temp_order)
                        elif stock['position'] < 0:
                           temp_order = ib_insync.MarketOrder('BUY', abs(stock['position']))
                           self.ib.placeOrder(temp_contract, temp_order)
                        #print('Close the positions')
                        self.holding_pairs = self.holding_pairs[(self.holding_pairs.first_stock == local_holding_pairs_first) & (self.holding_pairs.first_stock == local_holding_pairs_second)]
                        self.holding_pairs.to_csv('trades/holding_pairs.csv',index = False, header = True)
                        print('cleaned holding_pairs.csv')
            else:
                print('No Change for pairs/n')
            #print('pair : [%s,%s]\n'%(local_holding_pairs_first,local_holding_pairs_second))
            self.df = df
            print(df.tail()[['y','x','hr','zScore','numUnits']])


    def check_all_current_pairs (self): 
        is_empty, self.holding_pairs = self.get_local_position()
        ib_position_list = self.get_IB_position()
        print('now checking current pairs...')
        for pair in range(len(self.holding_pairs)):
            local_holding_pairs = self.holding_pairs
            local_holding_pairs_first = (local_holding_pairs['first_stock'].values[pair-1])
            local_holding_pairs_second = (local_holding_pairs['second_stock'].values[pair-1])
            local_holding_pairs = (local_holding_pairs_first,local_holding_pairs_second)

            self.first_inst.append(local_holding_pairs_first)
            self.second_inst.append(local_holding_pairs_second)

            ib_position_list_symbols = []
            ib_position_list_symbols = [i['symbol'] for i in ib_position_list]
            msg = "Holdin pairs stored in local drive:{}. \n All holding in IB: {}".format(local_holding_pairs,ib_position_list_symbols) 
            self._logging(msg)

            if set(local_holding_pairs).issubset(set(ib_position_list_symbols)):
                print('holding pairs stored in local drive found in all holding in IB: passed')
                df = self.compute_signal(local_holding_pairs_first,local_holding_pairs_second)
                self.df_list.append(df)
                changing_dict = utils.changing_pairs_dict(local_holding_pairs_first,local_holding_pairs_second,df = df)
                print (changing_dict)
                if changing_dict is not None: #Changed signal
                    if changing_dict['To_signal'] == 0 or df['numUnits'][-1] == 0:
                      print ('ib_position_list', ib_position_list_symbols)
                      for stock in ib_position_list:
                        if stock['symbol'] in local_holding_pairs:
                            print ('is in local_holding_pairs')
                            temp_contract = ib_insync.Stock(stock['symbol'],exchange='SMART',currency='USD')
                            self.ib.qualifyContracts(temp_contract)
                            print('Closing %s with %s'%(stock['symbol'],stock['position']))
                            if stock['position'] > 0: 
                                temp_order = ib_insync.MarketOrder('SELL', stock['position'])
                                self.ib.placeOrder(temp_contract, temp_order)
                            elif stock['position'] < 0:
                               temp_order = ib_insync.MarketOrder('BUY', abs(stock['position']))
                               self.ib.placeOrder(temp_contract, temp_order)
                        self.holding_pairs = self.holding_pairs[~((self.holding_pairs.first_stock == local_holding_pairs_first) & (self.holding_pairs.second_stock == local_holding_pairs_second))]
                        self.holding_pairs.to_csv('trades/holding_pairs.csv',index = False, header = True)
                        print('cleaned holding_pairs.csv')

                else:
                    print('No Change for pairs/n')
                #print('pair : [%s,%s]\n'%(local_holding_pairs_first,local_holding_pairs_second))
                print(df.tail()[['y','x','hr','zScore','numUnits']])
                self._logging(df.tail()[['y','x','hr','zScore','numUnits']])

        
    def run_strategy(self):
        ib_position_list = self.get_IB_position()
        ib_position_list_symbols = [i['symbol'] for i in ib_position_list]
        is_empty, get_local_position = self.get_local_position()
        current_no_pair = int(len(ib_position_list)/2)
        #if empty pairs or havn't filled all the required number of paris
        if is_empty or (current_no_pair < self.required_no_pairs):   
            self.find_changing_pairs()
            df,latest_date = self.load_latest_changing_pairs()
            can_trade = df[(df['To_signal']!=0)&(abs(df['Hedge Ratio'])>1)]
            if can_trade.empty:
                print ('Empty changing pair')
            else:
                while current_no_pair < self.required_no_pairs:
                    first_row = can_trade.iloc[current_no_pair]
                    pair = (first_row['first_stock'],first_row['second_stock'])
                    is_in_current_pairs = np.all([set(x).issubset(set(ib_position_list_symbols) ) for x in pair])
                    if is_in_current_pairs == False:
                        first_row = can_trade.iloc[current_no_pair+self.required_no_pairs]
                    x_stock = first_row['first_stock']
                    self.first_inst.append(x_stock)
                    y_stock = first_row['second_stock']
                    self.second_inst.append(y_stock)

                    ratio = abs(round(first_row['Hedge Ratio']))
                    x_df = self.get_data(x_stock,self.start,self.end)
                    y_df = self.get_data(y_stock,self.start,self.end)
                    common_index = (x_df.index).intersection(y_df.index)
                    if len(common_index) > self.len_constrain_day:
                        x_df = x_df.loc[common_index].copy()
                        y_df = y_df.loc[common_index].copy() 
                        df, sharpe, _ = kalman.backtest(x_df['Adj Close'], y_df['Adj Close']) #df1 = x; df2 = y
                        df = df.dropna()
                        df.to_csv('log/%s_%s.csv'%(x_stock,y_stock))
                        #self.df = df 

                        ##save trading record
                        dir = 'trades'
                        for f in os.listdir(dir):
                            os.remove(os.path.join(dir, f))
                        #first_row.to_frame().T.to_csv('trades/lastest_trading_pairs.csv')


                        if first_row['To_signal']==1:
                            #Long second (y); Short first (x) * hedge ratio

                            one_unit_cost = df['x'][-1]*ratio+df['y'][-1] 
                            current_spread = df['y'][-1]  - df['x'][-1]*ratio
                            one_unit_quantity = round(self.capital/one_unit_cost)

                            x_contract = ib_insync.Stock(x_stock,exchange='SMART',currency='USD')
                            self.ib.qualifyContracts(x_contract)
                            x_contract_quantity = one_unit_quantity*ratio
                            order1 = ib_insync.MarketOrder('SELL', x_contract_quantity)
                            self.ib.placeOrder(x_contract, order1)
                            print('Placed: SELL %s %s'%(x_stock,x_contract_quantity))

                            y_contract = ib_insync.Stock(y_stock,exchange='SMART',currency='USD')
                            self.ib.qualifyContracts(y_contract)
                            y_contract_quantity = one_unit_quantity
                            order2 = ib_insync.MarketOrder('BUY', y_contract_quantity)
                            self.ib.placeOrder(y_contract, order2)
                            print('Placed: BUY %s %s'%(y_stock,y_contract_quantity))

                            msg = 'Short %s: %s * %i ; Long %s: %s'%(x_stock,x_contract_quantity,ratio,y_stock,y_contract_quantity)
                            self._logging(msg)
                            print(msg)

                        elif first_row['To_signal']== -1:

                            #Short second (y); Long first (x) * hedge ratio
                            one_unit_cost = df['x'][-1]*ratio+df['y'][-1] 
                            current_spread = df['y'][-1]  - df['x'][-1]*ratio
                            one_unit_quantity = round(self.capital/one_unit_cost)

                            x_contract = ib_insync.Stock(x_stock,exchange='SMART',currency='USD')
                            self.ib.qualifyContracts(x_contract)
                            x_contract_quantity = one_unit_quantity*ratio
                            order1 = ib_insync.MarketOrder('BUY', x_contract_quantity)
                            self.ib.placeOrder(x_contract, order1)
                            print('Placed: BUY %s %s'%(x_stock,x_contract_quantity))

                            y_contract = ib_insync.Stock(y_stock,exchange='SMART',currency='USD')
                            self.ib.qualifyContracts(y_contract)
                            y_contract_quantity = one_unit_quantity
                            order2 = ib_insync.MarketOrder('SELL', y_contract_quantity)
                            self.ib.placeOrder(y_contract, order2)
                            print('Placed: SELL %s %s'%(y_stock,y_contract_quantity))

                            msg = 'Long %s: %s; Short %s: %s * %i'%(x_stock,x_contract_quantity,y_stock,y_contract_quantity,ratio)
                            self._logging(msg)
                            print(msg)

                        current_no_pair += 1
                        data = [[x_stock,x_contract_quantity,y_stock,y_contract_quantity,ratio]]
                        temp_holding_pairs = pd.DataFrame(data,columns = ['first_stock','first_quantity','second_stock','second_quantity','Hedge Ratio'])
                        print (temp_holding_pairs)
                        print ('saving to holding_pairs.csv.......')
                        self.holding_pairs = pd.concat([self.holding_pairs,temp_holding_pairs], ignore_index=True)
                        self.holding_pairs.to_csv('trades/holding_pairs.csv',index = False, header = True)
                        self.check_all_current_pairs()
                
        elif (current_no_pair == self.required_no_pairs):
            print('There are full trading pairs.')
            print ('Now checking the latest change of pairs')
            #last_changing = pd.read_csv('trades/lastest_trading_pairs.csv')
            self.check_all_current_pairs()
        else:
            print('some error')


from subprocess import Popen
import time

if __name__ == '__main__':
    p = Popen('StartTWS.bat',cwd = r"C:\IBC", shell = True)
    #p = Popen('StartGateway.bat',cwd = r"C:\IBC", shell = True)
    stdout, stderr = p.communicate()
    time.sleep(10)
    print("sleeping 10 seconds")

    date = dt.datetime.now()
    date ='%s_%s_%s'%(date.year,date.month,date.day)
    backtest_day = 730
    len_constrain_day =  365
    capital = 13000
    market = 'US'
    NO_OF_PAIRS = 3
    ACCOUNT_ID = 'U12191682'
    #market = 'HK'
    start = dt.datetime.now()- dt.timedelta(days=backtest_day)
    end = dt.datetime.now()
    #TypeOfFindingParis = 'allUS' #last_result or 'allUS' or 'allUS_NAS'
    TypeOfFindingParis = 'allUS_NAS'

    system = ib_pt(backtest_day= backtest_day, len_constrain_day= len_constrain_day, start_date = start, end_date = end
                     ,coint_pairs_type =TypeOfFindingParis , account = ACCOUNT_ID, capital = capital , no_pairs = NO_OF_PAIRS)
    system.run_strategy()
    system.close_ib()


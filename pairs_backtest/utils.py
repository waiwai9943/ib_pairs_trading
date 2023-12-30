from socket import gethostname
 #import email
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib
import json
from datetime import date
import pandas as pd
import os 
def send_email_pdf_figs(message):
    ## credits: http://linuxcursor.com/python-programming/06-how-to-send-pdf-ppt-attachment-with-html-body-in-python-script
    subject = 'Pairs Report of %s'%date.today()
    message = 'This is to send you the paris trading report of %s in HONG KONG \n %s \n '%(date.today(),message)

    reminder = '''
    \n
    ***********************************
    df1['spread'] = df1.y + (df1.x * df1.hr)
    Signal = -1 --> Short y, long x
    Signal = 0 --> Close Position
    Signal = 1 --> long y, Short x
    ***********************************
    '''

    message = message + reminder
    destination = 'waiwai9943@gmail.com'


    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login('waiwai9943@gmail.com', 'mzltdelgirssgqbk')
    # Craft message (obj)
    msg = MIMEMultipart()
    #message = f'{message}\nSend from Hostname: {gethostname()}'
    msg['Subject'] = subject
    msg['From'] = 'waiwai9943@gmail.com'
    msg['To'] = destination
    # Insert the text to the msg going by e-mail
    msg.attach(MIMEText(message, "plain"))
    # Attach the pdf to the msg going by e-mail
    #with open(re_outpdf, "rb") as f:
    #    #attach =
    #email.mime.application.MIMEApplication(f.read(),_subtype="pdf")
    #    attach = MIMEApplication(f.read(),_subtype="pdf")
    #attach.add_header('Content-Disposition','attachment',filename=str(re_outpdf))
    #msg.attach(attach)
    # send msg
    server.send_message(msg)


def changing_pairs_dict(first,second,df):
  if df.numUnits[-1]!=df.numUnits[-2]:
    temp_dict = {'first_stock':first,
                'second_stock':second,
                'from_date':'%s-%s-%s'%(df.index[-2].year,df.index[-2].month,df.index[-2].day),
                'from_signal':df.numUnits[-2],
                'To_date':'%s-%s-%s'%(df.index[-1].year,df.index[-1].month,df.index[-1].day),
                'To_signal':df.numUnits[-1],
                 'Hedge Ratio':df.hr[-1]}
    return temp_dict


def get_summary(path):
  date = dt.datetime.today().strftime('%Y_%m_%d')
  path = 'kalman_backtest_data_%s/'%date
  combined  =  pd.DataFrame()
  all_result = pd.DataFrame(columns = ["Pair", "cum rets"])
  for file in os.listdir(path):
    if file.endswith('.csv'): 
      df = pd.read_csv(file)
      last_row = df.tail(1)['cum rets']

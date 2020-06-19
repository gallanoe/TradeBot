import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

def MACD(df):
	print("Calculating MACD...")
	ema12 = df['Close'].ewm(span=12).mean()
	ema26 = df['Close'].ewm(span=26).mean()
	df['MACD'] = ema12 - ema26

def CCI(df):
	print("Calculating CCI...")
	tp = (df['High'] + df['Low'] + df['Close']) / 3
	atp = tp.rolling(window=20).mean()
	md = tp.rolling(window=20).apply(func=lambda x: np.fabs(x - x.mean()).mean(), raw=True)
	df['CCI'] = (tp - atp) / (0.015 * md)

def MA20TR(df):
  print("Calculating MA20TR...") 
  hl = df['High'] - df['Low']
  ch = df['Close'].shift(1) - df['High']
  cl = df['Close'].shift(1) - df['Low']

  tr = pd.concat([hl, ch, cl], keys=['HL', 'CH', 'CL'], axis=1)
  tr = tr.max(axis=1)
  df['MA20TR'] = tr.rolling(window=20).mean()

def BOLL(df):
  print("Calculating BOLU and BOLD...")
  tp = (df['High'] + df['Low'] + df['Close']) / 3
  mean = tp.rolling(window=20).mean()
  std = tp.rolling(window=20).std()
  df['BOLU'] = mean + 2 * std
  df['BOLD'] = mean - 2 * std

def MA(df):
  print("Calculating MA5, MA10, EMA20...")
  df['MA5'] = df['Close'].rolling(window=5).mean()
  df['MA10'] = df['Close'].rolling(window=10).mean()
  df['EMA20'] = df['Close'].ewm(span=20).mean()

def MTM(df):
  print("Calculating MTM6 and MTM12...") 
  df['MTM6'] = 100 * df['Close'] / df['Close'].shift(129)
  df['MTM12'] = 100 * df['Close'] / df['Close'].shift(258)

def ROC(df):
  print("Calculating ROC...")
  df['ROC'] = 100 * (df['Close'] - df['Close'].shift(12)) / df['Close']

def SMI(df):
  print("Calculating SMI...")
  hmax = df['High'].rolling(3).max()
  lmin = df['Low'].rolling(3).min()
  c = (hmax + lmin) / 2
  d = df['Close'] - c
  ds1 = d.ewm(span=3).mean()
  ds2 = ds1.ewm(span=3).mean()
  dhl1 = (hmax - lmin).ewm(span=3).mean()
  dhl2 = dhl1.ewm(span=3).mean() / 2
  df['SMI'] = 100 * (ds2 / dhl2)

def WVAD(df):
  print("Calculating WVAD...")
  ad = pd.Series(np.zeros(len(df.index)), index=df.index)
  trl = pd.concat([df['Low'], df['Close'].shift(1)], axis=1).min(axis=1)
  trh = pd.concat([df['High'], df['Close'].shift(1)], axis=1).max(axis=1)
  ad.loc[df['Close'] > df['Close'].shift(1)] = df['Close'] - trl
  ad.loc[df['Close'] < df['Close'].shift(1)] = df['Close'] - trh
  df['WVAD'] = ad.cumsum()

def RSI(df):
  print("Calculating RSI...")
  diff = df['Close'].diff()
  u, d = diff.copy(), diff.copy()
  u[u < 0] = 0
  d[d > 0] = 0 
  d = d.abs()
  u = u.ewm(alpha=(1/14)).mean()
  d = d.ewm(alpha=(1/14)).mean()
  df['RSI'] = 100 - (100 / (1 + (u / d)))

def RET(df):
  print("Calculating RET1, RET5, RET20...")
  df['RET1'] = df['Close']/df['Open']
  df['RET5'] = df['Close']/df['Open'].shift(5)
  df['RET10'] = df['Close']/df['Open'].shift(10)
  df['RET20'] = df['Close']/df['Open'].shift(20)

def VOL(df):
  print("Calculating VOL1, VOL5, VOL10, VOL20...")
  df['VOL1'] = df['Close'].rolling(2).std()
  df['VOL5'] = df['Close'].rolling(5).std()
  df['VOL10'] = df['Close'].rolling(10).std()
  df['VOL20'] = df['Close'].rolling(20).std()
  
def graph(df, columns, title):
  plt.figure(figsize=(14,7), dpi=100)
  for column in columns:
    plt.plot(df[column], label=column)
  plt.legend()
  plt.xlabel('Date')
  plt.ylabel('USD')
  plt.title(title)
  plt.show()
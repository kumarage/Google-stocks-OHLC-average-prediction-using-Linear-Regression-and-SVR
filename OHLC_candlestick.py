# graph plot help: https://pythonprogramming.net/candlestick-ohlc-graph-matplotlib-tutorial/
# gives a cool explanation
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import datetime as dt #to convert date into numerical format
import matplotlib.dates as mdates
#draw candlestick OHLC with matplotlib
import matplotlib.ticker as mticker
from matplotlib.finance import candlestick_ohlc


#create a dataframe df and get values in the csv file
filename='HistoricalQuotes.csv'
df = pd.read_csv(filename)

price_high = df['high']
price_low = df['low']
price_open = df['open']
price_close = df['close']
volume = df['volume']
dates = df['date']
#convert date format to numeric format
dates = pd.to_datetime(df['date'])
dates = dates.map(dt.datetime.toordinal)



fig = plt.figure()
ax1 = plt.subplot2grid((1,1),(0,0))

x = 0;
y = len(dates)
ohlc = []

while x < y:
	append_me = dates[x], price_open[x], price_high[x], price_low[x], price_close[x], volume[x]
	ohlc.append(append_me)
	x += 1


candlestick_ohlc(ax1,ohlc,width=0.5,colorup='g',colordown='b')

for label  in ax1.xaxis.get_ticklabels():
	label.set_rotation(45)

#convert date numeric to date format
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax1.xaxis.set_major_locator(mticker.MaxNLocator(8))

plt.title('OHLC variation of Google Stocks from 12/2018 to 3/2018')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

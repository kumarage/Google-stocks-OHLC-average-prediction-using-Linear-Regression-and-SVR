#sklearn google stock regression
#github help: https://github.com/chaitjo/regression-stock-prediction
#sklearn help : http://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn import svm
from sklearn.model_selection import train_test_split

import datetime as dt #to convert date into numerical format
import matplotlib.dates as mdates
#draw candlestick OHLC with matplotlib
import matplotlib.ticker as mticker
from matplotlib.finance import candlestick_ohlc

def regression_predict(dates_train, dates_test, OHLC_train):
	lin_model = linear_model.LinearRegression()
	lin_model.fit(dates_train,OHLC_train)
	predicted_price = lin_model.predict(dates_test)
	return predicted_price

def svm_predict(dates_train, dates_test, OHLC_train):
	
	SVR_rbf = svm.SVR(kernel='rbf', C=100, gamma=0.1)
	SVR_rbf.fit(dates_train,OHLC_train) 
	predict_rbf = SVR_rbf.predict(dates_test)

	SVR_sig = svm.SVR(kernel='sigmoid', C=1e3, gamma=0.1)
	SVR_sig.fit(dates_train,OHLC_train) 
	predict_sig = SVR_sig.predict(dates_test)

	#SVR_pol = svm.SVR(kernel='poly', C=1e3, degree=1)
	#SVR_pol.fit(dates_train,OHLC_train) 
	#predict_pol = SVR_pol.predict(dates_test)
	return predict_rbf , predict_sig #, predict_pol


#create a dataframe df and get values in the csv file
filename='HistoricalQuotes.csv'
df = pd.read_csv(filename)

price_high = df['high']
price_low = df['low']
price_open = df['open']
price_close = df['close']
#from these prices calculate the OHLC average for the Google stocks
price_ave= (price_high[:]+price_low[:]+price_open[:]+price_close[:])/4
#get dates from the dataframe
dates = df['date']
volume = df['volume']
#convert date format to numeric format
dates = pd.to_datetime(df['date'])
dates = dates.map(dt.datetime.toordinal)

#flatten an array into nX1
dates = dates.values.reshape(len(dates),1) # converting to matrix of nX1

#print averga OHLC values
#print(price_ave)

#split the data set in to train and test
dates_train, dates_test, OHLC_train, OHLC_test = train_test_split(dates,price_ave,
	test_size=0.25)

#for day in df['date']:

#call linear model to fit data into regression model and calculate for new data
pred_linear = regression_predict(dates_train, dates_test, OHLC_train)

#call svm regression model to fit data into 3 svr models and calculate for new data
pred_rbf, pred_sig = svm_predict(dates_train, dates_test, OHLC_train)

#plot all the results

plt.figure(1)
ax = plt.subplot2grid((1,1),(0,0))
plt.scatter(dates_test, OHLC_test, color='orange', label='data')
plt.plot(dates_test,pred_linear,color='black', label='Linear Regression')
plt.plot(dates_test,pred_rbf,color='yellow', label='SVR rbf')
plt.plot(dates_test,pred_sig,color='red', label='SVR sigmoid')
plt.xlabel('numeric dates')
plt.ylabel('target')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
plt.legend()
plt.show()

#draw candlestick OHLC with matplotlib
fig = plt.figure(2)
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
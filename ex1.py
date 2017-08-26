import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

df = quandl.get("WIKI/GOOGL")

print(df.head(10))

#############################################################################
#
#               Open    High     Low    Close      Volume  Ex-Dividend  \
# Date
# 2004-08-19  100.01  104.06   95.96  100.335  44659000.0          0.0
# 2004-08-20  101.01  109.08  100.50  108.310  22834300.0          0.0
# 2004-08-23  110.76  113.48  109.05  109.400  18256100.0          0.0
# 2004-08-24  111.24  111.60  103.57  104.870  15247300.0          0.0
# 2004-08-25  104.76  108.00  103.88  106.000   9188600.0          0.0
#
#             Split Ratio  Adj. Open  Adj. High   Adj. Low  Adj. Close  \
# Date
# 2004-08-19          1.0  50.159839  52.191109  48.128568   50.322842
# 2004-08-20          1.0  50.661387  54.708881  50.405597   54.322689
# 2004-08-23          1.0  55.551482  56.915693  54.693835   54.869377
# 2004-08-24          1.0  55.792225  55.972783  51.945350   52.597363
# 2004-08-25          1.0  52.542193  54.167209  52.100830   53.164113
#
#             Adj. Volume
# Date
# 2004-08-19   44659000.0
# 2004-08-20   22834300.0
# 2004-08-23   18256100.0
# 2004-08-24   15247300.0
# 2004-08-25    9188600.0
#
#############################################################################


# Making features
df['HL_PCT'] = ((df['Adj. High'] - df['Adj. Close'])/df['Adj. Close'])*100
df['PCT_change'] = ((df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open'])*100

# Final feature set
df = df[['HL_PCT', 'PCT_change', 'Adj. Close', 'Adj. Volume']]

forcast_column = 'Adj. Close'
df.fillna(-99999, inplace=True)  # make missing data an outlier
forecast_out = int(math.ceil(0.1*len(df)))  # 1% shift in data

df['label'] = df[forcast_column].shift(-forecast_out)

#############################################################################
#
#               HL_PCT  PCT_change  Adj. Close  Adj. Volume       label
# Date
# 2004-08-19  3.712563    0.324968   50.322842   44659000.0  214.973603
# 2004-08-20  0.710922    7.227007   54.322689   22834300.0  212.395645
# 2004-08-23  3.729433   -1.227880   54.869377   18256100.0  202.394773
# 2004-08-24  6.417469   -5.726357   52.597363   15247300.0  203.083148
# 2004-08-25  1.886792    1.183658   53.164113    9188600.0  207.686157
#
#############################################################################

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)  # normalize X

X_lately = X[-forecast_out:]
X = X[:-forecast_out:]

df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)  # making 20% testing data


LrModel = LinearRegression()  # n_jobs = 10 parameter means it will work 10 jobs parallel
LrModel.fit(X_train, y_train)
accuracy = LrModel.score(X_test, y_test)

# Every time we need a prediction we can not load full data and
# y_train a classifier therefore we use Pickle to dump the data
# in a file and use it in later time.

with open('linearregression ex1.pickle', 'wb') as f:
    pickle.dump(LrModel, f)

pickle_in = open('linearregression ex1.pickle', 'rb')
LrModel = pickle.load(pickle_in)

forecast_set = LrModel.predict(X_lately)

# print(forecast_set, accuracy, forecast_out)
# 0.97483414776

df['forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()



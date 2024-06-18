import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import pickle

df = pd.read_csv('home_prices.csv')
# plt.scatter(df.area,df.price)
# plt.xlabel('area')
# plt.ylabel('price (US$)')
# plt.show()

reg = linear_model.LinearRegression()
reg.fit(df[['area']].values, df.price)
print(reg.predict([[3300]]))
print(reg.coef_, reg.intercept_)

d = pd.read_csv('areas.csv')
d['prices'] = reg.predict(d.values)
d.to_csv("prediction.csv", index = False)

plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area, df.price, marker = '+')

plt.plot(df.area, reg.predict(df[['area']].values), color="blue")
# plt.show()

with open('model_pickle', 'wb') as f:
    pickle.dump(reg, f)

with open('model_pickle', 'rb') as f:
    model = pickle.load(f)

print(model.predict([[5000]]))

#if I wanted to use joblib
from sklearn.externals import joblib
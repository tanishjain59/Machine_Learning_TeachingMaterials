import pandas as pd
import numpy as np
from sklearn import linear_model

df = pd.read_csv('home_prices.csv')

import math
median_bedrooms = math.floor(df.bedrooms.median())
df.bedrooms = df.bedrooms.fillna(median_bedrooms)

#print(df)
reg = linear_model.LinearRegression()
reg.fit(df[['area', 'bedrooms', 'age']].values, df.price)

print(reg.predict([[3000, 3, 40]]))
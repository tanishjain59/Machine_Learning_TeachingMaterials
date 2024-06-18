import pandas as pd
df = pd.read_csv('ohe_house_prices.csv')

### DUMMY VARIABLE APPROACH ###
dummies = pd.get_dummies(df.town, dtype=int)
merged = pd.concat([df, dummies], axis = 'columns')
#need to drop the town column and one of the dummy variable columns
final = merged.drop(['town', 'west windsor'], axis = 'columns')
#print(final)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
X = final.drop(['price'], axis = 'columns')
Y = final.price
model.fit(X, Y)
print(model.predict([[2800, 0, 1]]))

#accuracy of model
print(model.score(X,Y))

###ONE HOT ENCODING APPROACH###

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dfle = df
dfle.town = le.fit_transform(dfle.town)
X = dfle[['town','area']].values
Y = dfle.price

""" NOT WORKING
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ohe = OneHotEncoder(categories=)
Xnew = ohe.fit_transform(X).toarray()
print(X)
print(Xnew)
Xnew = Xnew[:,1:]
#print(X)
#model.fit(X,Y)
#print(model.predict([[1,0,2800]]))
"""

